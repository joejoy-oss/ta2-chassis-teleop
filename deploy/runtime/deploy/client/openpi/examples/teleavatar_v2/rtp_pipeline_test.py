"""Tests for rtp_pipeline.py (no GStreamer needed), plus the layers that pass the decode settings down: their defaults,
and what main.py's options become at the RTP video interface (that one needs rclpy and gi, i.e. the client image)."""

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest
import rtp_pipeline  # pytest puts this directory on sys.path

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

# The receiver description of ade262e (rtp_video_interface._build_pipeline with its defaults), byte for byte, including
# the double space before "! queue".
GOLD_ADE262E = (
    'udpsrc port=8890 buffer-size=20000000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H265,'
    'payload=96" ! identity name=rtp_probe signal-handoffs=true silent=true ! rtph265depay ! h265parse '
    "config-interval=-1 disable-passthrough=true ! video/x-h265,stream-format=byte-stream,alignment=au ! identity "
    "name=h265_probe signal-handoffs=true silent=true ! nvh265dec max-display-delay=0 ! cudaconvert ! cudadownload "
    "! video/x-raw,format=RGB  ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! appsink "
    "name=decoded_frames emit-signals=true max-buffers=1 drop=true sync=false"
)
GOLD_SHA256 = "8419f36101052ed70adf600e20b1850f7d57c2e8b69da0739dd046df4e81f275"

DEFAULTS = {"port": 8890, "payload": 96, "udp_buffer_size": 20_000_000}

# The two queues the CPU and VA descriptions add, written out here rather than taken from rtp_pipeline, so that a change
# to them fails the tests.
RECEIVE_QUEUE = "queue name=receive_queue max-size-buffers=2048 max-size-bytes=0 max-size-time=0"
DECODE_QUEUE = "queue name=decode_queue max-size-buffers=16 max-size-bytes=0 max-size-time=0"


def test_default_description_is_byte_identical_to_ade262e():
    assert hashlib.sha256(GOLD_ADE262E.encode()).hexdigest() == GOLD_SHA256
    assert rtp_pipeline.build_pipeline_description(**DEFAULTS) == GOLD_ADE262E
    explicit = rtp_pipeline.build_pipeline_description(
        **DEFAULTS, decode_backend="nvidia", decoder="nvh265dec max-display-delay=0"
    )
    assert explicit == GOLD_ADE262E


def test_nvidia_ignores_cpu_settings_and_takes_the_decoder():
    with_cpu = rtp_pipeline.build_pipeline_description(
        **DEFAULTS, cpu_decode_threads=8, cpu_thread_type="frame", cpu_convert_threads=2
    )
    assert with_cpu == GOLD_ADE262E
    other = rtp_pipeline.build_pipeline_description(**DEFAULTS, decoder="nvh265dec")
    assert "! nvh265dec ! cudaconvert ! cudadownload ! video/x-raw,format=RGB  ! queue" in other


def test_cpu_description():
    description = rtp_pipeline.build_pipeline_description(**DEFAULTS, decode_backend="cpu")
    for nvidia_only in ("nvh265dec", "cuda"):
        assert nvidia_only not in description
    # The default description with the decoder swapped and the two CPU queues added, nothing else.
    edits = {
        'payload=96" ! ': f'payload=96" ! {RECEIVE_QUEUE} ! ',
        "name=h265_probe signal-handoffs=true silent=true ! ": (
            f"name=h265_probe signal-handoffs=true silent=true ! {DECODE_QUEUE} ! "
        ),
        "nvh265dec max-display-delay=0 ! cudaconvert ! cudadownload ! ": (
            "avdec_h265 max-threads=1 thread-type=auto ! videoconvert n-threads=4 ! "
        ),
    }
    expected = GOLD_ADE262E
    for old, new in edits.items():
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert description == expected


def test_cpu_settings_reach_the_description():
    description = rtp_pipeline.build_pipeline_description(
        port=18890,
        payload=97,
        udp_buffer_size=1000,
        decode_backend="cpu",
        cpu_decode_threads=0,
        cpu_thread_type="slice",
        cpu_convert_threads=2,
    )
    assert description.startswith("udpsrc port=18890 buffer-size=1000 ")
    assert "payload=97" in description
    assert "avdec_h265 max-threads=0 thread-type=slice ! videoconvert n-threads=2 !" in description
    assert rtp_pipeline.decode_fragment("cpu", cpu_thread_type="frame") == (
        "avdec_h265 max-threads=1 thread-type=frame ! videoconvert n-threads=4 ! video/x-raw,format=RGB"
    )


def test_va_description():
    description = rtp_pipeline.build_pipeline_description(**DEFAULTS, decode_backend="va")
    for other in ("nvh265dec", "cuda", "avdec_h265", "videoconvert"):
        assert other not in description
    # The CPU description's queues; the decoder named for the interface, its frames kept in VA memory (NV12).
    edits = {
        'payload=96" ! ': f'payload=96" ! {RECEIVE_QUEUE} ! ',
        "name=h265_probe signal-handoffs=true silent=true ! ": (
            f"name=h265_probe signal-handoffs=true silent=true ! {DECODE_QUEUE} ! "
        ),
        "nvh265dec max-display-delay=0 ! cudaconvert ! cudadownload ! video/x-raw,format=RGB ": (
            "vah265dec name=va_decoder ! video/x-raw(memory:VAMemory),format=NV12 "
        ),
    }
    expected = GOLD_ADE262E
    for old, new in edits.items():
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert description == expected
    other = rtp_pipeline.build_pipeline_description(**DEFAULTS, decode_backend="va", va_decoder="varenderD129h265dec")
    assert "! varenderD129h265dec name=va_decoder ! video/x-raw(memory:VAMemory),format=NV12  ! queue" in other
    # The VA settings do not touch the other backends.
    assert rtp_pipeline.build_pipeline_description(**DEFAULTS, va_decoder="varenderD129h265dec") == GOLD_ADE262E


@pytest.mark.parametrize(
    ("backend", "names"),
    [
        ("nvidia", ["rtp_probe", "h265_probe", "decoded_frames"]),
        ("cpu", ["receive_queue", "rtp_probe", "h265_probe", "decode_queue", "decoded_frames"]),
        ("va", ["receive_queue", "rtp_probe", "h265_probe", "decode_queue", "va_decoder", "decoded_frames"]),
    ],
)
def test_descriptions_keep_the_element_names_and_the_leaky_queue(backend, names):
    description = rtp_pipeline.build_pipeline_description(**DEFAULTS, decode_backend=backend)
    assert [token.split("=", 1)[1] for token in description.split() if token.startswith("name=")] == names
    parts = description.split(" ! ")
    elements = [part.split()[0] for part in parts]
    assert elements.count("queue") == 1 + sum(name.endswith("_queue") for name in names)
    # The unnamed queue right before the appsink is the leaky one (the interface counts its overruns).
    assert parts[-2] == "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0"
    assert elements[-1] == "appsink"
    known = set(rtp_pipeline.COMMON_ELEMENTS) | set(rtp_pipeline.BACKEND_ELEMENTS[backend])
    assert {e for e in elements if not e.startswith("video/")} <= known


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decode_backend": "vaapi"},
        {"decode_backend": "cpu", "cpu_thread_type": "slices"},
        {"decode_backend": "cpu", "cpu_decode_threads": -1},
        {"decode_backend": "cpu", "cpu_convert_threads": -1},
        {"decode_backend": "va", "va_decoder": ""},
        {"decode_backend": "va", "va_decoder": "vah265dec ! videoconvert"},
    ],
)
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):  # noqa: PT011
        rtp_pipeline.build_pipeline_description(**DEFAULTS, **kwargs)


@pytest.mark.parametrize(
    ("last", "seq", "expected"),
    [
        (None, 1234, (0, 0, 1234)),  # first packet
        (10, 11, (0, 0, 11)),
        (10, 14, (3, 0, 14)),  # 11, 12, 13 missing
        (65535, 0, (0, 0, 0)),  # wrap
        (65534, 1, (2, 0, 1)),  # 65535 and 0 missing across the wrap
        (10, 10, (0, 1, 10)),  # duplicate
        (10, 9, (0, 1, 10)),  # late
        (0, 65535, (0, 1, 0)),  # late across the wrap
        (0, 0x7FFF, (0x7FFE, 0, 0x7FFF)),  # largest forward jump
        (0, 0x8000, (0, 1, 0)),  # half the space away: treated as behind
    ],
)
def test_rtp_seq_step(last, seq, expected):
    assert rtp_pipeline.rtp_seq_step(last, seq) == expected


def test_rtp_seq_step_counts_a_lossy_sequence():
    gaps = out_of_order = 0
    last = None
    for seq in [65533, 65534, 0, 1, 1, 4, 3, 5]:  # 65535 lost, 1 duplicated, 2 lost, 3 late
        missing, late, last = rtp_pipeline.rtp_seq_step(last, seq)
        gaps += missing
        out_of_order += late
    assert (gaps, out_of_order, last) == (3, 2, 5)


def _class(path: pathlib.Path, name: str) -> ast.ClassDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {path.name}")


def _init_defaults(path: pathlib.Path, cls: str, names: tuple[str, ...]) -> dict:
    """Source text of the default values of the named parameters of cls.__init__."""
    init = next(n for n in _class(path, cls).body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    arguments = init.args.args + init.args.kwonlyargs
    defaults = [None] * (len(init.args.args) - len(init.args.defaults)) + init.args.defaults + init.args.kw_defaults
    return {a.arg: ast.unparse(d) for a, d in zip(arguments, defaults, strict=True) if a.arg in names}


def _field_defaults(path: pathlib.Path, cls: str, names: tuple[str, ...]) -> dict:
    """Source text of the default values of the named (annotated) fields of a dataclass."""
    return {
        item.target.id: ast.unparse(item.value)
        for item in _class(path, cls).body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id in names
    }


def test_layers_default_to_the_unchanged_nvidia_pipeline():
    """main.Args -> TeleavatarEnvironment -> TeleavatarROS2Interface -> RTPH265VideoInterface: without the new
    options every layer asks for the default (NVIDIA) description."""
    layer_names = (
        "video_decode",
        "video_cpu_threads",
        "video_cpu_thread_type",
        "video_convert_threads",
        "video_va_decoder",
    )
    expected = {
        "video_decode": "'nvidia'",
        "video_cpu_threads": "1",
        "video_cpu_thread_type": "'auto'",
        "video_convert_threads": "4",
        "video_va_decoder": "'vah265dec'",
    }
    assert _field_defaults(_HERE / "main.py", "Args", layer_names) == expected
    assert _init_defaults(_HERE / "env.py", "TeleavatarEnvironment", layer_names) == expected
    assert _init_defaults(_HERE / "ros2_interface.py", "TeleavatarROS2Interface", layer_names) == expected

    interface_names = (
        "decoder",
        "decode_backend",
        "cpu_decode_threads",
        "cpu_thread_type",
        "cpu_convert_threads",
        "va_decoder",
        "va_frame_size",
        "va_diagnostics",
    )
    assert _init_defaults(_HERE / "rtp_video_interface.py", "RTPH265VideoInterface", interface_names) == {
        "decoder": "rtp_pipeline.DEFAULT_NVIDIA_DECODER",
        "decode_backend": "rtp_pipeline.NVIDIA",
        "cpu_decode_threads": "1",
        "cpu_thread_type": "'auto'",
        "cpu_convert_threads": "4",
        "va_decoder": "rtp_pipeline.DEFAULT_VA_DECODER",
        "va_frame_size": "rtp_pipeline.DEFAULT_VA_FRAME_SIZE",
        "va_diagnostics": "False",
    }
    assert rtp_pipeline.NVIDIA == "nvidia"
    assert rtp_pipeline.DEFAULT_NVIDIA_DECODER == "nvh265dec max-display-delay=0"
    assert rtp_pipeline.DEFAULT_VA_DECODER == "vah265dec"
    assert rtp_pipeline.DEFAULT_VA_FRAME_SIZE == (1280, 2720)
    # main.py's choices for --video-decode.
    annotations = {
        item.target.id: ast.unparse(item.annotation)
        for item in _class(_HERE / "main.py", "Args").body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }
    assert annotations["video_decode"] == "Literal['nvidia', 'cpu', 'va']"
    assert annotations["video_va_decoder"] == "str"


# main.py's whole path down to the RTP video interface, in a fresh interpreter (main() sets signal handlers and runs
# rclpy): main.Args -> TeleavatarEnvironment -> TeleavatarROS2Interface -> RTPH265VideoInterface. The video interface is
# the real one, never started (no pipeline, no socket), and prints the settings it was built with; the ROS2 interface is
# the real one minus the robot (initial data at once, dummy observations); the policy is a stand-in.
_LAYERS_DRIVER = """
import json
import logging
import sys

import numpy as np

from examples.teleavatar_v2 import main as _main
from examples.teleavatar_v2 import ros2_interface as _ros2_interface


class Video(_ros2_interface.RTPH265VideoInterface):
    def __init__(self, **settings):
        super().__init__(**settings)
        print(json.dumps({"settings": settings, "description": self.pipeline_description}), flush=True)

    def start(self):
        pass


class Interface(_ros2_interface.TeleavatarROS2Interface):
    def wait_for_initial_data(self, timeout=10.0):
        return True

    def get_observation(self):
        image = np.zeros((4, 4, 3), np.uint8)
        images = dict.fromkeys(("left_color", "right_color", "head_camera"), image)
        return {"state": np.zeros(48, np.float32), "images": images}


class Policy:
    def __init__(self, host, port):
        pass

    def get_server_metadata(self):
        return {}

    def infer(self, obs):
        return {"actions": np.zeros((30, 16), np.float32)}

    def reset(self):
        pass


_ros2_interface.RTPH265VideoInterface = Video
_ros2_interface.TeleavatarROS2Interface = Interface
_main._websocket_client_policy.WebsocketClientPolicy = Policy
logging.basicConfig(level=logging.INFO, force=True)
sys.exit(_main.main(_main.Args(num_episodes=1, max_episode_steps=1, **json.loads(sys.argv[1]))))
"""

# What main.py's five video options become at the interface when none is given.
_INTERFACE_DEFAULTS = {
    "decode_backend": "nvidia",
    "cpu_decode_threads": 1,
    "cpu_thread_type": "auto",
    "cpu_convert_threads": 4,
    "va_decoder": "vah265dec",
}


@pytest.mark.parametrize(
    ("options", "interface", "fragment"),
    [
        pytest.param({}, {}, "! nvh265dec max-display-delay=0 ! cudaconvert ! cudadownload !", id="defaults"),
        pytest.param(
            {
                "video_decode": "cpu",
                "video_cpu_threads": 3,
                "video_cpu_thread_type": "slice",
                "video_convert_threads": 2,
                "video_va_decoder": "varenderD129h265dec",
            },
            {
                "decode_backend": "cpu",
                "cpu_decode_threads": 3,
                "cpu_thread_type": "slice",
                "cpu_convert_threads": 2,
                "va_decoder": "varenderD129h265dec",
            },
            "! avdec_h265 max-threads=3 thread-type=slice ! videoconvert n-threads=2 !",
            id="cpu",
        ),
        pytest.param(
            {"video_decode": "va", "video_va_decoder": "varenderD129h265dec"},
            {"decode_backend": "va", "va_decoder": "varenderD129h265dec"},
            "! varenderD129h265dec name=va_decoder ! video/x-raw(memory:VAMemory),format=NV12 ",
            id="va",
        ),
    ],
)
def test_main_passes_the_decode_settings_down_to_the_interface(options, interface, fragment):
    """B2 review J10: what main.Args is given reaches the RTP video interface through every layer, at run time
    (test_layers_default_to_the_unchanged_nvidia_pipeline only reads each layer's defaults). The cpu case gives all
    five options values other than their defaults, so that a layer passing on its own default instead fails it. The
    interface then describes the pipeline of exactly those settings; with no options, ade262e's byte for byte. Needs
    rclpy, gi and tyro (the client image)."""
    pytest.importorskip("rclpy")
    pytest.importorskip("gi")
    pytest.importorskip("tyro")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", _LAYERS_DRIVER, json.dumps(options)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    built = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(built) == 1, result.stdout
    expected = {**_INTERFACE_DEFAULTS, **interface}
    assert built[0]["settings"] == {"port": 8890, "payload": 96, **expected}
    assert built[0]["description"] == rtp_pipeline.build_pipeline_description(**DEFAULTS, **expected)
    assert fragment in built[0]["description"]
    if not options:
        assert built[0]["description"] == GOLD_ADE262E


def test_interface_uses_the_gold_description():
    """Where PyGObject/GStreamer are installed (the client image), the interface's own description."""
    pytest.importorskip("gi")
    import rtp_video_interface

    assert rtp_video_interface.RTPH265VideoInterface().pipeline_description == GOLD_ADE262E
    cpu = rtp_video_interface.RTPH265VideoInterface(decode_backend="cpu", cpu_decode_threads=2, port=18890)
    assert cpu.pipeline_description == rtp_pipeline.build_pipeline_description(
        **{**DEFAULTS, "port": 18890}, decode_backend="cpu", cpu_decode_threads=2
    )
    va = rtp_video_interface.RTPH265VideoInterface(decode_backend="va", va_decoder="varenderD129h265dec")
    assert va.pipeline_description == rtp_pipeline.build_pipeline_description(
        **DEFAULTS, decode_backend="va", va_decoder="varenderD129h265dec"
    )
    with pytest.raises(ValueError):  # noqa: PT011
        rtp_video_interface.RTPH265VideoInterface(decode_backend="gpu")


def test_interface_finds_the_leaky_queue():
    """start() counts the overruns of the queue feeding the appsink, also in the CPU description with its two other
    (non-leaky, bounded) queues."""
    pytest.importorskip("gi")
    import rtp_video_interface

    description = rtp_pipeline.build_pipeline_description(**DEFAULTS, decode_backend="cpu")
    pipeline = rtp_video_interface.Gst.parse_launch(description)
    leaky = rtp_video_interface._leaky_queue(pipeline.get_by_name("decoded_frames"))  # noqa: SLF001
    assert leaky.get_name() not in ("receive_queue", "decode_queue")
    assert (int(leaky.get_property("leaky")), leaky.get_property("max-size-buffers")) == (2, 1)  # 2: downstream
    for name, max_buffers in (("receive_queue", 2048), ("decode_queue", 16)):
        queue = pipeline.get_by_name(name)
        assert (int(queue.get_property("leaky")), queue.get_property("max-size-buffers")) == (0, max_buffers)
        assert (queue.get_property("max-size-bytes"), queue.get_property("max-size-time")) == (0, 0)


def test_overruns_are_split_at_the_start_of_the_stream(monkeypatch, caplog):
    """get_stats() times the overruns from the first frame handed over: up to STARTUP_OVERRUN_WINDOW_S they belong to
    the start of the stream (the acceptance criterion allows one there), later ones not; without a first frame they
    all count as later. Each of the first ones is logged."""
    pytest.importorskip("gi")
    import rtp_video_interface

    now = [100.0]
    monkeypatch.setattr(rtp_video_interface.time, "monotonic", lambda: now[0])
    interface = rtp_video_interface.RTPH265VideoInterface(decode_backend="cpu")
    interface._on_queue_overrun(None)  # noqa: SLF001
    stats = interface.get_stats()
    split = ("queue_overruns", "queue_overruns_startup", "queue_overruns_after_startup")
    assert stats["queue_overrun_times_s"] == []
    assert [stats[key] for key in split] == [1, 0, 1]
    interface._first_frame_at = 100.2  # noqa: SLF001  (as _image_callback sets it)
    for t in (100.9, 101.2, 101.3):
        now[0] = t
        interface._on_queue_overrun(None)  # noqa: SLF001
    stats = interface.get_stats()
    assert stats["queue_overrun_times_s"] == [-0.2, 0.7, 1.0, 1.1]
    assert [stats[key] for key in split] == [4, 3, 1]
    assert "1.100 s after the first frame" in caplog.text
    assert "before the first frame was handed over" in caplog.text


def test_va_conversion_failures_in_a_row_stop_the_stream():
    """A sample VaFrameTaker cannot convert is dropped; VA_MAX_FAILURES (3) of them in a row end the stream (the loop
    quits, stream_ended() turns true). A frame converted in between starts the count again."""
    pytest.importorskip("gi")
    import numpy as np
    import rtp_video_interface
    import va_frame

    class Loop:
        quits = 0

        def quit(self):
            self.quits += 1

    class Buffer:
        pts = 123

    class Sample:
        def get_buffer(self):
            return Buffer()

    class Sink:
        def emit(self, name):
            assert name == "pull-sample"
            return Sample()

    class Taker:
        def __init__(self, results):
            self.results = list(results)

        def take(self, _sample):
            if self.results.pop(0) == "fail":
                raise va_frame.VaFrameError("vapp_convert", -5018)
            return np.zeros((2720, 1280, 3), np.uint8)

    assert rtp_video_interface.VA_MAX_FAILURES == 3
    interface = rtp_video_interface.RTPH265VideoInterface(decode_backend="va")
    loop = interface._loop = Loop()  # noqa: SLF001  (as start() sets them)
    interface._va = Taker(["fail", "fail", "ok", "fail", "fail", "fail"])  # noqa: SLF001
    seen = []
    for _ in range(6):
        assert interface._image_callback(Sink()) == rtp_video_interface.Gst.FlowReturn.OK  # noqa: SLF001
        seen.append((interface._va_failures, interface.stream_ended(), loop.quits))  # noqa: SLF001
    assert seen == [(1, False, 0), (2, False, 0), (0, False, 0), (1, False, 0), (2, False, 0), (3, True, 1)]
    assert interface._frame_count == 1  # noqa: SLF001  (only the converted frame was handed over)


@pytest.mark.parametrize("fail", [None, "prepare", "ready", "open"])
def test_va_open_order_leaves_only_what_needs_the_display_between_ready_and_playing(monkeypatch, fail):
    """_open_va: the taker's prepare() (library, buffers, cv2) before READY, where udpsrc binds the port and nobody
    reads the socket until PLAYING; only open() (surfaces, context, warm-up) after it. Each failure releases what was
    made: a failed READY closes the prepared taker and sets the pipeline back to NULL."""
    pytest.importorskip("gi")
    import rtp_video_interface
    import va_frame

    Gst = rtp_video_interface.Gst  # noqa: N806
    events = []

    class Taker:
        def __init__(self, pipeline, **_kwargs):
            pass

        def prepare(self):
            events.append("prepare")
            if fail == "prepare":
                raise RuntimeError("no libvapp.so")

        def open(self):
            events.append("open")
            if fail == "open":
                raise va_frame.VaFrameError("vapp_create", -3001)
            return {"context": 8.0}

        def close(self):
            events.append("close")

    class Pipeline:
        def set_state(self, state):
            events.append(state.value_nick)
            if state == Gst.State.READY and fail == "ready":
                return Gst.StateChangeReturn.FAILURE
            return Gst.StateChangeReturn.SUCCESS

    monkeypatch.setattr(va_frame, "VaFrameTaker", Taker)
    interface = rtp_video_interface.RTPH265VideoInterface(decode_backend="va")
    if fail is None:
        assert isinstance(interface._open_va(Pipeline()), Taker)  # noqa: SLF001
        assert events == ["prepare", "ready", "open"]
        return
    with pytest.raises(RuntimeError, match="READY failed" if fail == "ready" else "GPU conversion"):
        interface._open_va(Pipeline())  # noqa: SLF001
    expected = {
        "prepare": ["prepare"],
        "ready": ["prepare", "ready", "close", "null"],
        "open": ["prepare", "ready", "open", "null"],
    }
    assert events == expected[fail]
