import threading
import time

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.sync.client

from openpi.serving import websocket_policy_server


class _SlowPolicy(_base_policy.BasePolicy):
    """Sleeps in `infer`, like a first request that compiles the model, and counts overlapping calls."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls = 0

    def infer(self, obs: dict) -> dict:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls += 1
        time.sleep(self._seconds)
        with self._lock:
            self._active -= 1
        return {"actions": np.zeros((2, 3)), "echo": obs["x"]}


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(policy: _base_policy.BasePolicy) -> int:
    port = _free_port()
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


def _connect(port: int, **kwargs) -> websockets.sync.client.ClientConnection:
    deadline = time.monotonic() + 10
    while True:
        try:
            ws = websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None, **kwargs)
            break
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    msgpack_numpy.unpackb(ws.recv())  # metadata
    return ws


def test_keepalive_is_answered_while_infer_runs():
    # The client pings every 0.1 s and closes the connection after 0.3 s without a pong; infer takes 1.5 s.
    port = _serve(_SlowPolicy(1.5))
    with _connect(port, ping_interval=0.1, ping_timeout=0.3) as ws:
        ws.send(msgpack_numpy.Packer().pack({"x": np.arange(3)}))
        result = msgpack_numpy.unpackb(ws.recv())
    np.testing.assert_array_equal(result["echo"], np.arange(3))
    assert result["server_timing"]["infer_ms"] >= 1500


def test_requests_from_two_connections_do_not_overlap():
    policy = _SlowPolicy(0.3)
    port = _serve(policy)
    echoes = []

    def client(i: int) -> None:
        with _connect(port) as ws:
            for _ in range(2):
                ws.send(msgpack_numpy.Packer().pack({"x": np.array([i])}))
                echoes.append(int(msgpack_numpy.unpackb(ws.recv())["echo"][0]))

    threads = [threading.Thread(target=client, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(echoes) == [0, 0, 1, 1]
    assert policy.calls == 4
    assert policy.max_active == 1
