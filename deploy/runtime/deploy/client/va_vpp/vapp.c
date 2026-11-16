/* vapp.c -> /opt/intel-va/lib/libvapp.so: VA-API video processing for the TeleAvatar V2 client (openpi-xpu-client).
 *
 * Converts frames that GStreamer 1.20's vah265dec decodes into VA memory (NV12 GstBuffers) to RGBX, written by the
 * GPU straight into memory the caller owns: every output slot is a user-pointer VA surface over a page-aligned
 * buffer of the caller (examples/teleavatar_v2/va_frame.py mmaps them). This skips GStreamer 1.20's slow download
 * path for VA memory.
 *
 * Colour: the input is converted as BT.709 limited range, whatever the stream's VUI says. That is what the original
 * client's NVIDIA (cudaconvert) and CPU (videoconvert) paths do with the robot's stream, which is flagged full range
 * without a colour description; the output is full-range RGB. The driver's fast path (VA_PROC_PIPELINE_FAST,
 * VA_FILTER_SCALING_FAST) is used when flags has VAPP_FAST: the default path's chroma is visibly worse.
 *
 *   int     vapp_abi(void)                                       VAPP_ABI; the Python side checks it
 *   vapp_t *vapp_create(decoder, width, height, dst, slots, pitch, fourcc, flags, &status)
 *           decoder: the vah265dec GstElement, from READY on (it opens its VADisplay in NULL->READY). The display is
 *           taken with a "gst.va.display.handle" context query on the decoder's pads and a reference to it is held
 *           until vapp_destroy, so tearing down the pipeline first is safe. dst[i] (4096-aligned, pitch * height
 *           bytes) backs slot i. NULL on failure, *status < 0.
 *   int     vapp_warmup(v)                                       one full conversion into every slot from a temporary
 *           NV12 surface: first use of the driver, the context and the user pages happens here, not on the first frame
 *   int     vapp_convert(v, buffer, slot)                        buffer: a GstBuffer of VA memory from the decoder;
 *           waits for the decoder's surface, converts it into slot and waits for the result; 0 or < 0
 *   int     vapp_resize(v, width, height, dst, pitch)            new size and buffers (same slots), then a warmup
 *   int     vapp_times(v, out, n)                                per-step times in ms (VAPP_T_*), returns VAPP_T_COUNT
 *   int     vapp_context_targets(v)                              1 if the driver needed the surfaces as render targets
 *   void    vapp_destroy(v)
 * Errors: -1 bad argument, -2 no VA display, -3 not a VA memory buffer, -(1000..7000) - VAStatus from the VA call of
 * that stage (config, surfaces, context, parameter buffer, render, input sync, warmup surface).
 * Threads: vapp_create, vapp_warmup and vapp_resize before or between conversions; vapp_convert from one thread at a
 * time; vapp_destroy after the last vapp_convert has returned.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gst/gst.h>
#include <va/va.h>
#include <va/va_vpp.h>

#define VAPP_ABI 1
#define VAPP_FAST 1
#define VAPP_MAX_SLOTS 8
/* GstVaMemory map flag (gst/va/gstvamemory.h from 1.22; private in 1.20): the mapped data is the VASurfaceID. */
#define GST_MAP_VA (GST_MAP_FLAG_LAST << 1)

/* vapp_times() indices: creation (config, context, surfaces), the last warmup, and the steps of the last conversion. */
enum {
    VAPP_T_CONFIG,
    VAPP_T_CONTEXT,
    VAPP_T_SURFACES,
    VAPP_T_WARMUP,
    VAPP_T_MAP,
    VAPP_T_SYNC_IN,
    VAPP_T_SUBMIT,
    VAPP_T_SYNC_OUT,
    VAPP_T_COUNT
};

typedef struct vapp {
    GstObject *display; /* the decoder's GstVaDisplay, referenced */
    VADisplay dpy;
    VAConfigID config;
    VAContextID context;
    int context_targets; /* 1: the driver wanted render targets, so the context goes with the surfaces */
    int width, height, pitch, slots, flags;
    unsigned int fourcc;
    VASurfaceID out[VAPP_MAX_SLOTS];
    double t[VAPP_T_COUNT];
} vapp_t;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static GstObject *display_of_context(GstContext *ctx)
{
    GstObject *display = NULL;
    if (ctx != NULL)
        gst_structure_get(gst_context_get_structure(ctx), "gst-display", GST_TYPE_OBJECT, &display, NULL);
    return display; /* referenced, or NULL */
}

/* The va decoder keeps its display to itself (it only posts it in a have-context message), so ask its pads with a
 * context query, which its query handlers answer from READY on; then try the element's own contexts. */
static GstObject *display_of_element(GstElement *element)
{
    static const char *pads[] = {"src", "sink"};
    GstObject *display = NULL;
    for (unsigned i = 0; i < 2 && display == NULL; i++) {
        GstPad *pad = gst_element_get_static_pad(element, pads[i]);
        GstQuery *q = gst_query_new_context("gst.va.display.handle");
        if (pad != NULL && gst_pad_query(pad, q)) {
            GstContext *ctx = NULL;
            gst_query_parse_context(q, &ctx);
            display = display_of_context(ctx);
        }
        gst_query_unref(q);
        if (pad != NULL)
            gst_object_unref(pad);
    }
    if (display == NULL) {
        GstContext *ctx = gst_element_get_context(element, "gst.va.display.handle");
        display = display_of_context(ctx);
        if (ctx != NULL)
            gst_context_unref(ctx);
    }
    return display;
}

static void destroy_surfaces(vapp_t *v)
{
    for (int i = 0; i < v->slots; i++) {
        if (v->out[i] != VA_INVALID_SURFACE)
            vaDestroySurfaces(v->dpy, &v->out[i], 1);
        v->out[i] = VA_INVALID_SURFACE;
    }
}

/* One user-pointer RGBX surface per slot over dst[i]. */
static int create_surfaces(vapp_t *v, void **dst)
{
    for (int i = 0; i < v->slots; i++) {
        uintptr_t ptr = (uintptr_t)dst[i];
        VASurfaceAttribExternalBuffers ext;
        VASurfaceAttrib attrs[3];
        VAStatus st;

        memset(&ext, 0, sizeof ext);
        ext.pixel_format = v->fourcc;
        ext.width = (uint32_t)v->width;
        ext.height = (uint32_t)v->height;
        ext.data_size = (uint32_t)v->pitch * (uint32_t)v->height;
        ext.num_planes = 1;
        ext.pitches[0] = (uint32_t)v->pitch;
        ext.buffers = &ptr;
        ext.num_buffers = 1;
        memset(attrs, 0, sizeof attrs);
        attrs[0].type = VASurfaceAttribPixelFormat;
        attrs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
        attrs[0].value.type = VAGenericValueTypeInteger;
        attrs[0].value.value.i = (int)v->fourcc;
        attrs[1].type = VASurfaceAttribMemoryType;
        attrs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
        attrs[1].value.type = VAGenericValueTypeInteger;
        attrs[1].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_USER_PTR;
        attrs[2].type = VASurfaceAttribExternalBufferDescriptor;
        attrs[2].flags = VA_SURFACE_ATTRIB_SETTABLE;
        attrs[2].value.type = VAGenericValueTypePointer;
        attrs[2].value.value.p = &ext;
        st = vaCreateSurfaces(v->dpy, VA_RT_FORMAT_RGB32, (unsigned)v->width, (unsigned)v->height, &v->out[i], 1,
                              attrs, 3);
        if (st != VA_STATUS_SUCCESS) {
            v->out[i] = VA_INVALID_SURFACE;
            return -2000 - (int)st;
        }
    }
    return 0;
}

static int create_context_with_targets(vapp_t *v)
{
    VAStatus st = vaCreateContext(v->dpy, v->config, v->width, v->height, VA_PROGRESSIVE, v->out, v->slots,
                                  &v->context);
    if (st != VA_STATUS_SUCCESS) {
        v->context = VA_INVALID_ID;
        return -3000 - (int)st;
    }
    return 0;
}

static int check_buffers(void **dst, int slots, int width, int height, int pitch)
{
    if (dst == NULL || slots < 1 || slots > VAPP_MAX_SLOTS || width <= 0 || height <= 0 || pitch < width * 4)
        return -1;
    for (int i = 0; i < slots; i++)
        if (dst[i] == NULL || ((uintptr_t)dst[i] & 4095) != 0)
            return -1;
    return 0;
}

int vapp_abi(void)
{
    return VAPP_ABI;
}

void vapp_destroy(vapp_t *v)
{
    if (v == NULL)
        return;
    if (v->dpy != NULL) {
        if (v->context != VA_INVALID_ID)
            vaDestroyContext(v->dpy, v->context);
        destroy_surfaces(v);
        if (v->config != VA_INVALID_ID)
            vaDestroyConfig(v->dpy, v->config);
    }
    if (v->display != NULL)
        gst_object_unref(v->display);
    free(v);
}

vapp_t *vapp_create(GstElement *decoder, int width, int height, void **dst, int slots, int pitch, unsigned int fourcc,
                    int flags, int *status)
{
    vapp_t *v;
    VAStatus st;
    double t0;
    int rc;

    if (status == NULL)
        return NULL;
    *status = -1;
    if (!GST_IS_ELEMENT(decoder) || check_buffers(dst, slots, width, height, pitch) != 0)
        return NULL;
    v = calloc(1, sizeof *v);
    if (v == NULL)
        return NULL;
    v->config = VA_INVALID_ID;
    v->context = VA_INVALID_ID;
    for (int i = 0; i < VAPP_MAX_SLOTS; i++)
        v->out[i] = VA_INVALID_SURFACE;
    v->width = width;
    v->height = height;
    v->pitch = pitch;
    v->slots = slots;
    v->fourcc = fourcc;
    v->flags = flags;

    v->display = display_of_element(decoder);
    if (v->display != NULL)
        g_object_get(v->display, "va-display", &v->dpy, NULL);
    if (v->dpy == NULL) {
        *status = -2;
        vapp_destroy(v);
        return NULL;
    }

    t0 = now_ms();
    st = vaCreateConfig(v->dpy, VAProfileNone, VAEntrypointVideoProc, NULL, 0, &v->config);
    v->t[VAPP_T_CONFIG] = now_ms() - t0;
    if (st != VA_STATUS_SUCCESS) {
        v->config = VA_INVALID_ID;
        *status = -1000 - (int)st;
        vapp_destroy(v);
        return NULL;
    }
    /* A video processing context without render targets, as GStreamer's own VA filter creates it: it does not depend
     * on the surfaces, so a resize keeps it. If the driver refuses, the context is created with the surfaces. */
    t0 = now_ms();
    st = vaCreateContext(v->dpy, v->config, 0, 0, 0, NULL, 0, &v->context);
    v->t[VAPP_T_CONTEXT] = now_ms() - t0;
    if (st != VA_STATUS_SUCCESS) {
        v->context = VA_INVALID_ID;
        v->context_targets = 1;
    }
    t0 = now_ms();
    rc = create_surfaces(v, dst);
    v->t[VAPP_T_SURFACES] = now_ms() - t0;
    if (rc == 0 && v->context_targets) {
        t0 = now_ms();
        rc = create_context_with_targets(v);
        v->t[VAPP_T_CONTEXT] += now_ms() - t0;
    }
    if (rc != 0) {
        *status = rc;
        vapp_destroy(v);
        return NULL;
    }
    *status = 0;
    return v;
}

/* One conversion of surface `in` into slot `slot` (the caller has waited for `in`). */
static int process(vapp_t *v, VASurfaceID in, int slot)
{
    VAProcPipelineParameterBuffer p;
    VABufferID pb;
    VAStatus st;
    double t0 = now_ms();

    memset(&p, 0, sizeof p);
    p.surface = in;
    p.surface_color_standard = VAProcColorStandardBT709;
    p.input_color_properties.color_range = VA_SOURCE_RANGE_REDUCED;
    p.input_color_properties.chroma_sample_location = VA_CHROMA_SITING_VERTICAL_CENTER | VA_CHROMA_SITING_HORIZONTAL_LEFT;
    p.output_color_standard = VAProcColorStandardBT709;
    p.output_color_properties.color_range = VA_SOURCE_RANGE_FULL;
    if (v->flags & VAPP_FAST) {
        p.pipeline_flags = VA_PROC_PIPELINE_FAST;
        p.filter_flags = VA_FILTER_SCALING_FAST;
    }
    st = vaCreateBuffer(v->dpy, v->context, VAProcPipelineParameterBufferType, sizeof p, 1, &p, &pb);
    if (st != VA_STATUS_SUCCESS)
        return -4000 - (int)st;
    st = vaBeginPicture(v->dpy, v->context, v->out[slot]);
    if (st == VA_STATUS_SUCCESS)
        st = vaRenderPicture(v->dpy, v->context, &pb, 1);
    if (st == VA_STATUS_SUCCESS)
        st = vaEndPicture(v->dpy, v->context);
    v->t[VAPP_T_SUBMIT] = now_ms() - t0;
    t0 = now_ms();
    if (st == VA_STATUS_SUCCESS)
        st = vaSyncSurface(v->dpy, v->out[slot]);
    v->t[VAPP_T_SYNC_OUT] = now_ms() - t0;
    vaDestroyBuffer(v->dpy, pb);
    return st == VA_STATUS_SUCCESS ? 0 : -5000 - (int)st;
}

int vapp_warmup(vapp_t *v)
{
    VASurfaceID tmp = VA_INVALID_SURFACE;
    VAStatus st;
    double t0 = now_ms();
    int rc = 0;

    if (v == NULL)
        return -1;
    st = vaCreateSurfaces(v->dpy, VA_RT_FORMAT_YUV420, (unsigned)v->width, (unsigned)v->height, &tmp, 1, NULL, 0);
    if (st != VA_STATUS_SUCCESS)
        return -7000 - (int)st;
    for (int i = 0; i < v->slots && rc == 0; i++)
        rc = process(v, tmp, i);
    vaDestroySurfaces(v->dpy, &tmp, 1);
    v->t[VAPP_T_WARMUP] = now_ms() - t0;
    return rc;
}

int vapp_convert(vapp_t *v, GstBuffer *buffer, int slot)
{
    GstMemory *mem;
    GstMapInfo info;
    VASurfaceID in = VA_INVALID_SURFACE;
    VAStatus st;
    double t0 = now_ms();

    if (v == NULL || slot < 0 || slot >= v->slots)
        return -1;
    if (!GST_IS_BUFFER(buffer) || gst_buffer_n_memory(buffer) < 1)
        return -3;
    mem = gst_buffer_peek_memory(buffer, 0);
    if (mem != NULL && gst_memory_map(mem, &info, GST_MAP_READ | GST_MAP_VA)) {
        in = *(VASurfaceID *)info.data;
        gst_memory_unmap(mem, &info);
    }
    v->t[VAPP_T_MAP] = now_ms() - t0;
    if (in == VA_INVALID_SURFACE)
        return -3;
    /* vah265dec pushes its surfaces without waiting for the decode to finish. */
    t0 = now_ms();
    st = vaSyncSurface(v->dpy, in);
    v->t[VAPP_T_SYNC_IN] = now_ms() - t0;
    if (st != VA_STATUS_SUCCESS)
        return -6000 - (int)st;
    return process(v, in, slot);
}

int vapp_resize(vapp_t *v, int width, int height, void **dst, int pitch)
{
    double t0;
    int rc;

    if (v == NULL || check_buffers(dst, v->slots, width, height, pitch) != 0)
        return -1;
    if (v->context_targets && v->context != VA_INVALID_ID) {
        vaDestroyContext(v->dpy, v->context);
        v->context = VA_INVALID_ID;
    }
    destroy_surfaces(v);
    v->width = width;
    v->height = height;
    v->pitch = pitch;
    t0 = now_ms();
    rc = create_surfaces(v, dst);
    v->t[VAPP_T_SURFACES] = now_ms() - t0;
    if (rc == 0 && v->context_targets) {
        t0 = now_ms();
        rc = create_context_with_targets(v);
        v->t[VAPP_T_CONTEXT] = now_ms() - t0;
    }
    return rc != 0 ? rc : vapp_warmup(v);
}

int vapp_times(const vapp_t *v, double *out, int n)
{
    if (v == NULL || out == NULL)
        return -1;
    for (int i = 0; i < n && i < VAPP_T_COUNT; i++)
        out[i] = v->t[i];
    return VAPP_T_COUNT;
}

int vapp_context_targets(const vapp_t *v)
{
    return v == NULL ? -1 : v->context_targets;
}
