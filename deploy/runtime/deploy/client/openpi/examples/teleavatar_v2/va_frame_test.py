"""Tests for va_frame.py without a GPU, GStreamer or libvapp.so: the helpers, the buffers, the ABI check, and
VaFrameTaker's slot rotation, resize and error paths against a fake library."""

import ctypes

import numpy as np
import pytest
import va_frame  # pytest puts this directory on sys.path


def test_fourcc_and_status_texts():
    assert va_frame.fourcc("RGBX") == 0x58424752
    with pytest.raises(ValueError):  # noqa: PT011
        va_frame.fourcc("RGB")
    assert va_frame.describe_status(-1) == "bad argument"
    assert "READY" in va_frame.describe_status(-2)
    assert "VA memory" in va_frame.describe_status(-3)
    assert va_frame.describe_status(-1001) == "vaCreateConfig (VideoProc) failed, VAStatus 0x1"
    assert va_frame.describe_status(-5018).startswith("video processing")
    assert va_frame.describe_status(-6002).startswith("vaSyncSurface on the decoded frame")
    assert va_frame.describe_status(-7003).endswith("VAStatus 0x3")
    error = va_frame.VaFrameError("vapp_convert", -5018)
    assert error.code == -5018
    assert "vapp_convert" in str(error)
    assert "0x12" in str(error)


def test_strip_x():
    pytest.importorskip("cv2")
    rgbx = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    rgb = va_frame.strip_x(rgbx)
    assert rgb.shape == (2, 3, 3)
    assert rgb.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(rgb, rgbx[..., :3])


def test_buffers_are_page_aligned_resident_and_closable():
    buffers = va_frame._Buffers(16, 8, 2)  # noqa: SLF001
    assert buffers.pitch == 64
    assert [a.shape for a in buffers.arrays] == [(8, 16, 4), (8, 16, 4)]
    assert all(int(a.sum()) == 0 for a in buffers.arrays)
    pointers = list(buffers.pointers)
    assert len(set(pointers)) == 2
    assert all(p % 4096 == 0 for p in pointers)
    kept = buffers.arrays[1][:1]  # a view left elsewhere (e.g. in a traceback) does not make close() fail
    buffers.close()
    assert buffers.arrays == []
    assert int(kept.sum()) == 0  # its mapping stays until the view goes


class _Function:
    def __init__(self, implementation):
        self.implementation = implementation
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


class _FakeLibrary:
    """libvapp.so's functions: records the calls, converts nothing."""

    _name = "fake-libvapp.so"

    def __init__(self, abi=va_frame.ABI, convert_status=0, resize_status=0):
        self.vapp_abi = _Function(lambda: abi)
        self.vapp_create = _Function(self._create)
        self.vapp_warmup = _Function(lambda handle: 0)
        self.vapp_convert = _Function(lambda handle, buffer, slot: convert_status)
        self.vapp_resize = _Function(lambda handle, width, height, dst, pitch: resize_status)
        self.vapp_times = _Function(self._times)
        self.vapp_context_targets = _Function(lambda handle: 0)
        self.vapp_destroy = _Function(lambda handle: None)

    @staticmethod
    def _create(element, width, height, dst, slots, pitch, fourcc, flags, status):
        status._obj.value = 0  # noqa: SLF001  (ctypes.byref)
        return 1234

    @staticmethod
    def _times(handle, out, n):
        for i in range(n):
            out[i] = float(i)
        return n


class _Caps:
    def __init__(self, width, height):
        self.values = {"width": width, "height": height}

    def get_structure(self, _index):
        return self

    def get_value(self, name):
        return self.values[name]


class _Sample:
    def __init__(self, width=1280, height=2720):
        self._caps = _Caps(width, height)
        self._buffer = object()

    def get_caps(self):
        return self._caps

    def get_buffer(self):
        return self._buffer


class _Pipeline:
    def get_by_name(self, name):
        return object() if name == "va_decoder" else None


@pytest.fixture
def fake(monkeypatch):
    pytest.importorskip("cv2")
    library = _FakeLibrary()
    monkeypatch.setattr(va_frame, "load_library", lambda path=None: library)
    monkeypatch.setattr(va_frame, "_element_pointer", lambda element: 42)
    return library


def test_abi_mismatch_is_refused(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda path: _FakeLibrary(abi=va_frame.ABI + 1))
    with pytest.raises(RuntimeError, match="ABI"):
        va_frame.load_library("/nowhere/libvapp.so")


def test_open_take_rotates_slots_and_records_steps(fake):
    taker = va_frame.VaFrameTaker(_Pipeline(), frame_size=(64, 32), slots=2, record_steps=True)
    opened = taker.open()
    assert set(opened) == {"buffers", "config", "context", "surfaces", "warmup", "strip_warmup", "context_with_targets"}
    (element, width, height, _dst, slots, pitch, fourcc, flags, _status) = fake.vapp_create.calls[0]
    assert (element, width, height, slots, pitch, fourcc, flags) == (42, 64, 32, 2, 256, va_frame.fourcc("RGBX"), 1)
    frames = [taker.take(_Sample(64, 32)) for _ in range(3)]
    assert [call[2] for call in fake.vapp_convert.calls] == [0, 1, 0]
    assert all(frame.shape == (32, 64, 3) for frame in frames)
    steps = taker.steps()
    assert set(steps) == set(va_frame.STEPS)
    assert all(len(values) == 3 for values in steps.values())
    assert steps["sync_in"] == [5.0, 5.0, 5.0]  # vapp_times index 5
    taker.close()
    taker.close()
    assert len(fake.vapp_destroy.calls) == 1
    with pytest.raises(RuntimeError, match="not open"):
        taker.take(_Sample(64, 32))


def test_a_new_size_rebuilds_the_surfaces_once(fake):
    taker = va_frame.VaFrameTaker(_Pipeline(), frame_size=(64, 32), slots=2)
    taker.open()
    frame = taker.take(_Sample(64, 48))
    assert frame.shape == (48, 64, 3)
    assert taker.resizes == 1
    assert [call[1:3] for call in fake.vapp_resize.calls] == [(64, 48)]
    taker.take(_Sample(64, 48))
    assert taker.resizes == 1
    assert taker.steps()["strip"] == []  # record_steps off
    taker.close()


def test_failures_raise_with_the_status(fake):
    taker = va_frame.VaFrameTaker(_Pipeline(), frame_size=(64, 32))
    taker.open()
    fake.vapp_convert.implementation = lambda handle, buffer, slot: -5018
    with pytest.raises(va_frame.VaFrameError) as info:
        taker.take(_Sample(64, 32))
    assert info.value.code == -5018
    # A failed resize gives up the instance: later frames are refused, not converted into freed memory.
    fake.vapp_resize.implementation = lambda handle, width, height, dst, pitch: -2002
    with pytest.raises(va_frame.VaFrameError):
        taker.take(_Sample(64, 48))
    assert len(fake.vapp_destroy.calls) == 1
    with pytest.raises(RuntimeError, match="not open"):
        taker.take(_Sample(64, 32))
    taker.close()
    assert len(fake.vapp_destroy.calls) == 1


@pytest.mark.parametrize("where", ["element", "create", "warmup", "strip"])
def test_a_failed_open_releases_the_instance_and_the_buffers(fake, monkeypatch, where):
    def fail(*_args):
        raise RuntimeError(f"{where} failed")

    if where == "element":
        monkeypatch.setattr(va_frame, "_element_pointer", fail)
    elif where == "create":
        fake.vapp_create.implementation = lambda *_args: 0  # NULL, status left 0
    elif where == "warmup":
        fake.vapp_warmup.implementation = lambda handle: -7001
    else:
        monkeypatch.setattr(va_frame, "strip_x", fail)
    taker = va_frame.VaFrameTaker(_Pipeline(), frame_size=(64, 32))
    with pytest.raises(RuntimeError, match=where):  # the failure itself, not one from the cleanup (VaFrameError is one)
        taker.open()
    assert taker._handle is None  # noqa: SLF001
    assert taker._buffers is None  # noqa: SLF001
    assert len(fake.vapp_destroy.calls) == (1 if where == "warmup" else 0)  # the strip warm-up is in prepare() now
    with pytest.raises(RuntimeError, match="not open"):
        taker.take(_Sample(64, 32))
    taker.close()
    assert len(fake.vapp_destroy.calls) == (1 if where == "warmup" else 0)


def test_prepare_does_everything_that_needs_no_display_and_open_reuses_it(fake, monkeypatch):
    """prepare() (run before READY) maps the buffers and warms cv2 up without creating the instance; a second call
    does nothing; open() keeps the same buffers and creates the instance."""
    events = []
    strip = va_frame.strip_x
    create = fake.vapp_create.implementation

    def strip_and_note(rgbx):
        events.append("strip")
        return strip(rgbx)

    def create_and_note(*args):
        events.append("create")
        return create(*args)

    monkeypatch.setattr(va_frame, "strip_x", strip_and_note)
    fake.vapp_create.implementation = create_and_note
    taker = va_frame.VaFrameTaker(_Pipeline(), frame_size=(64, 32), slots=2)
    taker.prepare()
    buffers = taker._buffers  # noqa: SLF001
    assert buffers is not None
    assert events == ["strip"]
    taker.prepare()
    assert taker._buffers is buffers  # noqa: SLF001
    assert events == ["strip"]
    opened = taker.open()
    assert events == ["strip", "create"]
    assert taker._buffers is buffers  # noqa: SLF001
    assert fake.vapp_create.calls[0][3] is buffers.pointers
    assert set(opened) == {"buffers", "config", "context", "surfaces", "warmup", "strip_warmup", "context_with_targets"}
    taker.close()
    assert len(fake.vapp_destroy.calls) == 1


def test_bad_arguments():
    with pytest.raises(ValueError):  # noqa: PT011
        va_frame.VaFrameTaker(_Pipeline(), frame_size=(0, 32))
    with pytest.raises(ValueError):  # noqa: PT011
        va_frame.VaFrameTaker(_Pipeline(), slots=9)
    with pytest.raises(RuntimeError, match="no element"):
        va_frame.VaFrameTaker(_Pipeline(), element_name="missing").open()
