"""GStreamer pipeline descriptions and RTP bookkeeping for the TeleAvatar V2 RTP/H.265 camera stream.

No GStreamer import here, so this can be used and tested anywhere. RTPH265VideoInterface (rtp_video_interface.py)
launches the descriptions; the tools in rtp_tools/ reuse them.

Three decode backends:
- "nvidia" (default): nvh265dec, then NV12 -> RGB on the GPU (cudaconvert) and a single download. With the
  default arguments the description is byte for byte the one the interface always used.
- "cpu": avdec_h265 (gst-libav), then videoconvert. For machines without an NVIDIA GPU, such as the B70 server.
  libav's frame threading holds back one frame per extra thread (about 22 ms at 45 fps), so the default is a
  single decode thread. thread_type "slice" only helps when the stream is encoded with WPP (wavefront parallel
  processing, entropy_coding_sync_enabled_flag): without it libavcodec decodes the slices or tiles of a picture one
  after another.
  The CPU description also receives, parses and decodes on separate threads (CPU_RECEIVE_QUEUE, CPU_DECODE_QUEUE).
- "va": vah265dec (GStreamer's VA-API plugin; the B70's Intel GPU), with the CPU description's two queues. Its frames
  stay in VA memory (NV12) up to the appsink; the interface converts each one to RGB on the GPU, written straight into
  its own memory (va_frame.py). The decoder is named VA_DECODER_NAME so that the interface can find it.
"""

from __future__ import annotations

NVIDIA = "nvidia"
CPU = "cpu"
VA = "va"
DECODE_BACKENDS = (NVIDIA, CPU, VA)
DEFAULT_NVIDIA_DECODER = "nvh265dec max-display-delay=0"
# With one render node mapped into the container the VA plugin registers it as vah265dec; with several, the others get
# the node in the name (e.g. varenderD129h265dec).
DEFAULT_VA_DECODER = "vah265dec"
VA_DECODER_NAME = "va_decoder"
# The TeleAvatar V2 composite frame (width, height): the VA output surfaces are made this size before the first frame.
DEFAULT_VA_FRAME_SIZE = (1280, 2720)
CPU_THREAD_TYPES = ("auto", "frame", "slice")
# Element factories each backend adds to the common receiver elements.
COMMON_ELEMENTS = ("udpsrc", "identity", "rtph265depay", "h265parse", "queue", "appsink")
BACKEND_ELEMENTS = {
    NVIDIA: ("nvh265dec", "cudaconvert", "cudadownload"),
    CPU: ("avdec_h265", "videoconvert"),
    VA: (DEFAULT_VA_DECODER,),
}
# CPU and VA descriptions. Decoding in udpsrc's thread let the socket buffer overflow while a slow frame (a keyframe)
# was being decoded; after the receive queue that thread only reads the socket. The decode queue gives the decoder
# its own thread and sits after h265_probe, so the time an access unit waits for the decoder counts in
# decode_latency. Both are bounded: under a sustained overload the backlog stops growing and the socket drops.
CPU_RECEIVE_QUEUE = "queue name=receive_queue max-size-buffers=2048 max-size-bytes=0 max-size-time=0"
CPU_DECODE_QUEUE = "queue name=decode_queue max-size-buffers=16 max-size-bytes=0 max-size-time=0"


def decode_fragment(
    decode_backend: str = NVIDIA,
    *,
    decoder: str = DEFAULT_NVIDIA_DECODER,
    cpu_decode_threads: int = 1,
    cpu_thread_type: str = "auto",
    cpu_convert_threads: int = 4,
    va_decoder: str = DEFAULT_VA_DECODER,
) -> str:
    """Elements from the H.265 decoder to its output, without a leading or trailing "!": RGB in system memory for
    "nvidia" and "cpu", NV12 in VA memory for "va".

    `decoder` is the NVIDIA decoder element (with its properties); the other backends ignore it.
    `cpu_decode_threads` is avdec_h265's max-threads (0 = libav's automatic choice).
    `va_decoder` is the VA decoder element (vah265dec, or varenderD1xxh265dec when several render nodes are mapped).
    """
    if decode_backend == NVIDIA:
        # GPU NV12->RGB in CUDA memory + single download; ~3x lower latency than CPU videoconvert.
        return f"{decoder} ! cudaconvert ! cudadownload ! video/x-raw,format=RGB"
    if decode_backend == CPU:
        if cpu_thread_type not in CPU_THREAD_TYPES:
            raise ValueError(f"cpu_thread_type must be one of {CPU_THREAD_TYPES}, got {cpu_thread_type!r}")
        if cpu_decode_threads < 0 or cpu_convert_threads < 0:
            raise ValueError("thread counts must be >= 0")
        return (
            f"avdec_h265 max-threads={cpu_decode_threads} thread-type={cpu_thread_type} "
            f"! videoconvert n-threads={cpu_convert_threads} ! video/x-raw,format=RGB"
        )
    if decode_backend == VA:
        if not va_decoder.strip() or "!" in va_decoder:
            raise ValueError(f"va_decoder must be one element, got {va_decoder!r}")
        # Without the explicit VA memory caps, GStreamer 1.20's decoder negotiates system memory with the caps-less
        # appsink and downloads every frame (slow).
        return f"{va_decoder} name={VA_DECODER_NAME} ! video/x-raw(memory:VAMemory),format=NV12"
    raise ValueError(f"decode_backend must be one of {DECODE_BACKENDS}, got {decode_backend!r}")


def build_pipeline_description(
    *,
    port: int,
    payload: int,
    udp_buffer_size: int,
    decode_backend: str = NVIDIA,
    decoder: str = DEFAULT_NVIDIA_DECODER,
    cpu_decode_threads: int = 1,
    cpu_thread_type: str = "auto",
    cpu_convert_threads: int = 4,
    va_decoder: str = DEFAULT_VA_DECODER,
) -> str:
    """The receiver: UDP/RTP -> H.265 access units -> decode -> RGB (VA: NV12 in VA memory) -> appsink "decoded_frames".

    The two identity elements are probes: "rtp_probe" sees every RTP packet, "h265_probe" every access unit
    on its way to the decoder (the start of decode_latency). The queue before the appsink keeps only the newest frame.
    The CPU and VA descriptions add CPU_RECEIVE_QUEUE right after udpsrc and CPU_DECODE_QUEUE right after h265_probe.
    """
    caps = f"application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,payload={payload}"
    fragment = decode_fragment(
        decode_backend,
        decoder=decoder,
        cpu_decode_threads=cpu_decode_threads,
        cpu_thread_type=cpu_thread_type,
        cpu_convert_threads=cpu_convert_threads,
        va_decoder=va_decoder,
    )
    receive_queue = decode_queue = ""
    if decode_backend in (CPU, VA):
        receive_queue, decode_queue = f"! {CPU_RECEIVE_QUEUE} ", f"! {CPU_DECODE_QUEUE} "
    decode = (
        f'udpsrc port={port} buffer-size={udp_buffer_size} caps="{caps}" '
        f"{receive_queue}"
        "! identity name=rtp_probe signal-handoffs=true silent=true "
        "! rtph265depay "
        "! h265parse config-interval=-1 disable-passthrough=true "
        "! video/x-h265,stream-format=byte-stream,alignment=au "
        "! identity name=h265_probe signal-handoffs=true silent=true "
        f"{decode_queue}"
        f"! {fragment} "
    )
    appsink = (
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
        "! appsink name=decoded_frames emit-signals=true max-buffers=1 drop=true sync=false"
    )
    return f"{decode} ! {appsink}"


def rtp_seq_step(last: int | None, seq: int) -> tuple[int, int, int | None]:
    """Account for RTP sequence number `seq` arriving after `last` (16-bit, wrapping; None before the first packet).

    Returns (missing, out_of_order, new_last). A forward jump of n counts n - 1 missing packets. A packet at or behind
    `last` (late or duplicate) counts as out of order and leaves `last` unchanged.
    """
    if last is None:
        return 0, 0, seq
    delta = (seq - last) & 0xFFFF
    if delta == 0 or delta >= 0x8000:
        return 0, 1, last
    return delta - 1, 0, seq
