"""Tests for rtpcap.py: the file format, the packet digest, RFC 4571 reading and the synthesized send times."""

from __future__ import annotations

import io
import json
import struct

import pytest
import rtpcap  # pytest puts this directory on sys.path


def _packet(seq: int, ts: int, size: int = 20) -> bytes:
    return struct.pack("!BBHII", 0x80, 96, seq & 0xFFFF, ts & 0xFFFFFFFF, 1) + bytes(size - 12)


def test_round_trip(tmp_path):
    path = tmp_path / "x.rtpcap"
    records = [(10, _packet(1, 0)), (2**63 + 5, _packet(2, 0, 1400)), (11, b""), (12, bytes(range(256)) * 100)]
    meta = {"source": "test", "port": 18890, "note": "ünïcode"}
    with rtpcap.Writer(path, meta) as writer:
        for time_ns, packet in records:
            writer.write(time_ns, memoryview(packet))
    assert writer.packets == 4
    assert path.read_bytes().startswith(rtpcap.MAGIC)
    assert rtpcap.read_meta(path) == meta
    assert list(rtpcap.iter_packets(path)) == records
    with rtpcap.Reader(path) as reader:
        assert reader.meta["note"] == "ünïcode"
        assert list(reader) == records
    assert rtpcap.file_digest(path) == {
        "packets": 4,
        "packets_sha256": rtpcap.packets_sha256(packet for _t, packet in records),
    }


def test_truncated_and_foreign_files(tmp_path):
    path = tmp_path / "x.rtpcap"
    with rtpcap.Writer(path, {}) as writer:
        writer.write(1, _packet(1, 0))
        writer.write(2, _packet(2, 0))
    data = path.read_bytes()
    for cut in (1, 20, 25):  # inside the last packet, all of it, inside its record header
        (tmp_path / "cut.rtpcap").write_bytes(data[:-cut])
        with pytest.raises(ValueError, match="truncated"):
            list(rtpcap.iter_packets(tmp_path / "cut.rtpcap"))
    (tmp_path / "other").write_bytes(b"\x00\x00\x00\x01" + data)
    with pytest.raises(ValueError, match="not an rtpcap"):
        rtpcap.read_meta(tmp_path / "other")
    (tmp_path / "noline").write_bytes(rtpcap.MAGIC + b'{"a": 1}')
    with pytest.raises(ValueError, match="metadata"):
        rtpcap.read_meta(tmp_path / "noline")


def test_packets_sha256_counts_boundaries_not_times():
    assert rtpcap.packets_sha256([b"ab", b"c"]) != rtpcap.packets_sha256([b"a", b"bc"])
    assert rtpcap.packets_sha256([b"ab", b"c"]) != rtpcap.packets_sha256([b"c", b"ab"])
    assert rtpcap.packets_sha256([]) == rtpcap.packets_sha256(iter([]))
    assert len(rtpcap.packets_sha256([b"x"])) == 64


def test_rfc4571():
    packets = [_packet(1, 0), b"", bytes(1400)]
    stream = b"".join(len(p).to_bytes(2, "big") + p for p in packets)
    assert list(rtpcap.iter_rfc4571(io.BytesIO(stream))) == packets
    with pytest.raises(ValueError, match="truncated"):
        list(rtpcap.iter_rfc4571(io.BytesIO(stream[:-1])))
    with pytest.raises(ValueError, match="truncated"):
        list(rtpcap.iter_rfc4571(io.BytesIO(stream + b"\x05")))


def test_synthesized_send_times():
    # GStreamer's 45 fps timestamps step by 1999, 2000 or 2001 ticks; here they also wrap past 2^32.
    first = 2**32 - 3000
    timestamps = [first, first, first + 1999, first + 3999, first + 3999, first + 3999, first + 6000]
    sizes = [100, 1400, 1400, 1400, 1400, 50, 1400]
    packets = [_packet(i, ts, size) for i, (ts, size) in enumerate(zip(timestamps, sizes, strict=True))]
    times = [t for t, _p in rtpcap.synthesize_send_times(packets, fps=45, line_rate_bps=1e9, start_ns=1000)]
    frame = 1e9 / 45
    assert times == [
        1000,
        1000 + 800,  # 100 bytes at 1 Gbit/s
        1000 + round(frame),
        1000 + round(2 * frame),
        1000 + round(2 * frame) + 11_200,
        1000 + round(2 * frame) + 22_400,
        1000 + round(3 * frame),
    ]


def test_synthesized_send_times_never_go_back():
    # 2.8 Mbit in one access unit takes 2.8 ms at 1 Gbit/s; at 1 Mbit/s the next access unit has to wait.
    packets = [_packet(i, 2000 * (i // 2), 1400) for i in range(4)]
    times = [t for t, _p in rtpcap.synthesize_send_times(packets, fps=45, line_rate_bps=1e6)]
    assert times == [0, 11_200_000, 22_400_000, 33_600_000]
    assert times == sorted(times)


def test_from_rfc4571_command(tmp_path):
    packets = [_packet(i, 2000 * (i // 3), 700) for i in range(9)]
    (tmp_path / "in.rtpstream").write_bytes(b"".join(len(p).to_bytes(2, "big") + p for p in packets))
    out = tmp_path / "out.rtpcap"
    rtpcap.main(
        [
            "from-rfc4571",
            str(tmp_path / "in.rtpstream"),
            str(out),
            "--meta",
            "wpp=1",
            "--meta",
            "x265_options=ctu=32:wpp=1",
            "--meta",
            'layout={"head": [960, 960]}',
        ]
    )
    meta = rtpcap.read_meta(out)
    assert (meta["clock"], meta["send_model"]["fps"], meta["send_model"]["line_rate_bps"]) == (
        rtpcap.CLOCK_SYNTHETIC,
        45.0,
        1e9,
    )
    assert (meta["wpp"], meta["x265_options"], meta["layout"]) == (1, "ctu=32:wpp=1", {"head": [960, 960]})
    records = list(rtpcap.iter_packets(out))
    assert [p for _t, p in records] == packets
    assert [t for t, _p in records][::3] == [0, round(1e9 / 45), round(2e9 / 45)]


def test_digest_command(tmp_path, capsys):
    path = tmp_path / "x.rtpcap"
    with rtpcap.Writer(path, {}) as writer:
        writer.write(5, _packet(1, 0))
    rtpcap.main(["digest", str(path)])
    line = json.loads(capsys.readouterr().out)
    assert line == {"file": str(path), "packets": 1, "packets_sha256": rtpcap.packets_sha256([_packet(1, 0)])}
