"""RGB frames from vah265dec's VA memory, converted on the GPU straight into this process's memory.

GStreamer 1.20 can only download VA memory to system memory slowly. Here libvapp.so (C, openpi-xpu-env
client/va_vpp/vapp.c; /opt/intel-va/lib/libvapp.so in the client image) does the conversion with VA-API video
processing: each decoded NV12 surface becomes RGBX, written by the GPU into page-aligned buffers mapped here (one
user-pointer VA surface per slot). Stripping the X byte (cv2) gives the RGB frame. Colour: BT.709 limited range in,
full range out (see vapp.c), as the original client's NVIDIA and CPU paths convert the robot's stream.

    taker = VaFrameTaker(pipeline, frame_size=(1280, 2720))
    taker.prepare()                                           # library, buffers, cv2: before READY (see prepare())
    taker.open()                                              # the pipeline at READY: the decoder has its VADisplay;
                                                              # surfaces, context, one warm-up conversion per slot
    frame = taker.take(sample)                                # appsink callback: a new (H, W, 3) uint8 RGB array
    taker.close()                                             # after the pipeline is at NULL

The decoder element is found by name (rtp_pipeline.VA_DECODER_NAME). Needs gi (only to take the element's pointer),
numpy and cv2; importing this module needs neither gi nor the library.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import mmap
import os
import threading
import time

import numpy as np

ABI = 1  # vapp_abi() of the libvapp.so this module was written for
DEFAULT_LIB = "/opt/intel-va/lib/libvapp.so"
LIB_ENV = "TA2_VAPP_LIB"  # overrides DEFAULT_LIB (development)
FLAG_FAST = 1  # vapp.c VAPP_FAST: the driver's fast path
# vapp_times() order: creation, the last warm-up, then the steps of the last conversion.
TIMES = ("config", "context", "surfaces", "warmup", "map", "sync_in", "submit", "sync_out")
STEPS = ("map", "sync_in", "submit", "sync_out", "strip")  # per frame; strip is RGBX -> RGB here
_STAGES = (
    (-7000, "vaCreateSurfaces for the warm-up"),
    (-6000, "vaSyncSurface on the decoded frame"),
    (-5000, "video processing (render or vaSyncSurface on the output)"),
    (-4000, "vaCreateBuffer for the pipeline parameters"),
    (-3000, "vaCreateContext"),
    (-2000, "vaCreateSurfaces over the user memory"),
    (-1000, "vaCreateConfig (VideoProc)"),
)


def fourcc(code: str) -> int:
    """VA fourcc of a four-letter code, e.g. "RGBX"."""
    if len(code) != 4:
        raise ValueError(f"fourcc needs 4 characters, got {code!r}")
    return ord(code[0]) | ord(code[1]) << 8 | ord(code[2]) << 16 | ord(code[3]) << 24


def describe_status(code: int) -> str:
    """Readable form of a negative vapp status."""
    if code == -1:
        return "bad argument"
    if code == -2:
        return "no VA display from the decoder (is the pipeline at READY, with a render node?)"
    if code == -3:
        return "the sample is not a VA memory buffer"
    for base, what in _STAGES:
        if code <= base:
            return f"{what} failed, VAStatus {base - code:#x}"
    return f"status {code}"


class VaFrameError(RuntimeError):
    def __init__(self, what: str, code: int):
        super().__init__(f"{what}: {describe_status(code)} ({code})")
        self.code = code


def strip_x(rgbx: np.ndarray) -> np.ndarray:
    """(H, W, 4) RGBX -> a new contiguous (H, W, 3) RGB array."""
    import cv2  # only needed once frames flow

    return cv2.cvtColor(rgbx, cv2.COLOR_RGBA2RGB)


class _Buffers:
    """`slots` anonymous mappings of pitch * height bytes (page-aligned, as user-pointer surfaces need), written once so
    their pages are resident before the GPU pins them. close() unmaps them; one that still has a view somewhere (e.g. in
    the traceback of a failed open()) is unmapped when that view goes."""

    def __init__(self, width: int, height: int, slots: int):
        self.width, self.height, self.pitch = width, height, width * 4
        self._maps = [mmap.mmap(-1, self.pitch * height) for _ in range(slots)]
        self.arrays = [np.frombuffer(m, dtype=np.uint8).reshape(height, width, 4) for m in self._maps]
        for array in self.arrays:
            array.fill(0)
        self.pointers = (ctypes.c_void_p * slots)(*(a.ctypes.data for a in self.arrays))

    def close(self) -> None:
        self.arrays = []  # the views must go before the mappings can close
        for m in self._maps:
            with contextlib.suppress(BufferError):  # "cannot close exported pointers exist": left to the last view
                m.close()
        self._maps = []


def _element_pointer(element) -> int:
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return get(element.__gpointer__, None)


def load_library(path: str | None = None) -> ctypes.CDLL:
    """libvapp.so with its prototypes, after checking its ABI."""
    lib = ctypes.CDLL(path or os.environ.get(LIB_ENV) or DEFAULT_LIB)
    lib.vapp_abi.restype = ctypes.c_int
    lib.vapp_abi.argtypes = []
    if lib.vapp_abi() != ABI:
        raise RuntimeError(f"{lib._name} has ABI {lib.vapp_abi()}, this module needs {ABI}")  # noqa: SLF001
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    for name, restype, argtypes in (
        (
            "vapp_create",
            c_ptr,
            [c_ptr, c_int, c_int, ctypes.POINTER(c_ptr), c_int, c_int, ctypes.c_uint, c_int, ctypes.POINTER(c_int)],
        ),
        ("vapp_warmup", c_int, [c_ptr]),
        ("vapp_convert", c_int, [c_ptr, c_ptr, c_int]),
        ("vapp_resize", c_int, [c_ptr, c_int, c_int, ctypes.POINTER(c_ptr), c_int]),
        ("vapp_times", c_int, [c_ptr, ctypes.POINTER(ctypes.c_double), c_int]),
        ("vapp_context_targets", c_int, [c_ptr]),
        ("vapp_destroy", None, [c_ptr]),
    ):
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


class VaFrameTaker:
    """Converts the decoder's VA memory samples to RGB through libvapp.so (see the module docstring).

    frame_size (width, height) sizes the surfaces at open(), before any caps are known; a sample of another size makes
    take() rebuild them once (logged; that frame takes longer). slots output buffers are used in turn. With
    record_steps, the per-frame step times (STEPS, ms) are kept for steps().
    """

    def __init__(
        self,
        pipeline,
        *,
        element_name: str = "va_decoder",
        frame_size: tuple[int, int] = (1280, 2720),
        slots: int = 2,
        lib_path: str | None = None,
        record_steps: bool = False,
    ):
        self._pipeline = pipeline
        self._element_name = element_name
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        if self._frame_size[0] <= 0 or self._frame_size[1] <= 0 or not 1 <= slots <= 8:
            raise ValueError(f"bad frame_size {frame_size} or slots {slots}")
        self._slots = slots
        self._lib_path = lib_path
        self._record_steps = record_steps
        self._lib = None
        self._handle = None
        self._buffers: _Buffers | None = None
        self._prepare_ms: dict[str, float] = {}
        self._slot = 0
        self._lock = threading.Lock()
        self._times = (ctypes.c_double * len(TIMES))()
        self.open_ms: dict[str, float] = {}
        self.resizes = 0
        self._steps: dict[str, list[float]] = {name: [] for name in STEPS}
        self.logger = logging.getLogger(self.__class__.__name__)

    def prepare(self) -> None:
        """The part of open() that needs no VA display: load the library, map the buffers, and import cv2 with a first
        RGBX -> RGB (the import alone takes about 0.3 s). Call it before the pipeline goes to READY: udpsrc binds the
        port there and nothing reads the socket until PLAYING, so the time in between fills the socket buffer and then
        costs packets. open() calls it if it has not been called. A failure releases the buffers."""
        if self._buffers is not None:
            return
        self._lib = load_library(self._lib_path)
        t0 = time.perf_counter()
        self._buffers = _Buffers(*self._frame_size, self._slots)
        self._prepare_ms = {"buffers": (time.perf_counter() - t0) * 1e3}
        try:
            t0 = time.perf_counter()
            strip_x(self._buffers.arrays[0])  # imports cv2 and runs its first conversion here, not on the first frame
            self._prepare_ms["strip_warmup"] = (time.perf_counter() - t0) * 1e3
        except BaseException:
            self.close()
            raise

    def open(self) -> dict[str, float]:
        """Create the surfaces and the context and warm them up (prepare() first, if it has not been called). Returns
        the times (ms) of each part. Whatever fails after the buffers are mapped, the instance and the buffers are
        released before the exception propagates."""
        element = self._pipeline.get_by_name(self._element_name)
        if element is None:
            raise RuntimeError(f"no element named {self._element_name!r} in the pipeline")
        self.prepare()
        try:
            status = ctypes.c_int(0)
            handle = self._lib.vapp_create(
                _element_pointer(element),
                self._buffers.width,
                self._buffers.height,
                self._buffers.pointers,
                self._slots,
                self._buffers.pitch,
                fourcc("RGBX"),
                FLAG_FAST,
                ctypes.byref(status),
            )
            if not handle:
                raise VaFrameError("vapp_create", status.value or -1)
            self._handle = handle
            rc = self._lib.vapp_warmup(handle)
            if rc != 0:
                raise VaFrameError("vapp_warmup", rc)
            times = self._read_times()
        except BaseException:
            self.close()
            raise
        self.open_ms = {
            "buffers": self._prepare_ms["buffers"],  # mmap + writing every page (prepare())
            "config": times["config"],
            "context": times["context"],
            "surfaces": times["surfaces"],  # user-pointer surfaces
            "warmup": times["warmup"],  # one conversion per slot, first use of the context and the pages
            "strip_warmup": self._prepare_ms["strip_warmup"],  # cv2 import and first RGBX -> RGB (prepare())
            "context_with_targets": float(self._lib.vapp_context_targets(handle)),
        }
        return dict(self.open_ms)

    def take(self, sample) -> np.ndarray:
        """The sample's frame as a new contiguous RGB array (raises VaFrameError when the conversion fails)."""
        with self._lock:
            if self._handle is None:
                raise RuntimeError("VaFrameTaker is not open")
            structure = sample.get_caps().get_structure(0)
            width, height = int(structure.get_value("width")), int(structure.get_value("height"))
            if (width, height) != (self._buffers.width, self._buffers.height):
                self._resize(width, height)
            slot = self._slot
            rc = self._lib.vapp_convert(self._handle, hash(sample.get_buffer()), slot)  # hash() is the GstBuffer*
            if rc != 0:
                raise VaFrameError("vapp_convert", rc)
            t0 = time.perf_counter()
            frame = strip_x(self._buffers.arrays[slot])
            strip_ms = (time.perf_counter() - t0) * 1e3
            self._slot = (slot + 1) % self._slots
            if self._record_steps:
                times = self._read_times()
                for name in STEPS[:-1]:
                    self._steps[name].append(times[name])
                self._steps["strip"].append(strip_ms)
            return frame

    def steps(self) -> dict[str, list[float]]:
        """Per-frame step times in ms (record_steps only), in the order frames were taken."""
        return {name: list(values) for name, values in self._steps.items()}

    def close(self) -> None:
        """Destroy the surfaces and the context. Safe to call more than once and after the pipeline is at NULL (the
        library keeps its own reference to the decoder's display)."""
        with self._lock:
            if self._handle is not None:
                self._lib.vapp_destroy(self._handle)
                self._handle = None
            if self._buffers is not None:
                self._buffers.close()
                self._buffers = None

    def _resize(self, width: int, height: int) -> None:
        self.logger.warning(
            "VA frames are %dx%d, not %dx%d: rebuilding the output surfaces (this frame is late)",
            width,
            height,
            self._buffers.width,
            self._buffers.height,
        )
        buffers = _Buffers(width, height, self._slots)
        rc = self._lib.vapp_resize(self._handle, width, height, buffers.pointers, buffers.pitch)
        if rc != 0:
            # The library may hold surfaces over either set of buffers now: give up the instance before freeing them.
            self._lib.vapp_destroy(self._handle)
            self._handle = None
            buffers.close()
            self._buffers.close()
            self._buffers = None
            raise VaFrameError("vapp_resize", rc)
        self._buffers.close()
        self._buffers = buffers
        self._slot = 0
        self.resizes += 1

    def _read_times(self) -> dict[str, float]:
        self._lib.vapp_times(self._handle, self._times, len(TIMES))
        return dict(zip(TIMES, self._times, strict=True))
