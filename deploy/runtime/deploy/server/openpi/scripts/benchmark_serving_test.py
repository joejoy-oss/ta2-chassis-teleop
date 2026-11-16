import pathlib
import socket
import threading

import numpy as np
from openpi_client import base_policy as _base_policy
import pytest

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

from . import benchmark_serving
from . import serve_policy


class _ChunkPolicy(_base_policy.BasePolicy):
    """Answers like an RTC policy (a new chunk id per request) and records the RTC fields of each request."""

    def __init__(self):
        self.rtc_fields: list[dict] = []

    def infer(self, obs: dict) -> dict:
        self.rtc_fields.append({k: int(v) for k, v in obs.items() if k.startswith("rtc/")})
        return {
            "actions": np.zeros((30, 16)),
            _policy.RTC_CHUNK_ID: len(self.rtc_fields),
            "policy_timing": {"infer_ms": 1.0, "input_transform_ms": 0.5},
        }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("rtc", [False, True])
def test_client_continues_the_previous_chunk_and_records_each_timing_layer(rtc):
    policy = _ChunkPolicy()
    port = _free_port()
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    args = benchmark_serving.Args(
        out=pathlib.Path("unused.json"),
        policy=serve_policy.Checkpoint(config="pi05_teleavatar_v2", dir="unused"),
        rtc=serve_policy.RTCArgs(enabled=rtc),
        port=port,
    )
    client = benchmark_serving._Client(args)  # noqa: SLF001
    observation = benchmark_serving.make_observation("model", np.random.default_rng(0), args.prompt)
    record: dict = {}

    benchmark_serving._timed(client, observation, 3, record)  # noqa: SLF001

    if rtc:
        # The first request starts a chunk sequence; each later one continues the chunk the server returned last.
        steady = {
            _policy.RTC_PREFIX_START: args.prefix_start,
            _policy.RTC_INFERENCE_DELAY: args.inference_delay,
            _policy.RTC_EXECUTION_HORIZON: args.execution_horizon,
        }
        assert policy.rtc_fields == [
            {},
            {_policy.RTC_PREV_CHUNK_ID: 1, **steady},
            {_policy.RTC_PREV_CHUNK_ID: 2, **steady},
        ]
    else:
        assert policy.rtc_fields == [{}, {}, {}]
    assert len(record["client_ms"]) == 3
    assert len(record["server_infer_ms"]) == 3
    assert record["policy_infer_ms"] == [1.0] * 3
    assert record["policy_input_transform_ms"] == [0.5] * 3
    # The client round trip contains the server's infer time.
    assert all(c >= s for c, s in zip(record["client_ms"], record["server_infer_ms"], strict=True))
