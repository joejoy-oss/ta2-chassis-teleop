"""Tests for stream_info.py on small streams built from real x265 parameter sets and slice headers (see
h265_rtp_test.py): .rtpcap and Annex-B input, the bitstream summary, --drop-incomplete-aus (packets lost or late inside
an access unit and at its boundaries, a missing marker, no slice), picture reordering, and the command line."""

from __future__ import annotations

import json

import h265_rtp
from h265_rtp_test import MICRO_PPS
from h265_rtp_test import MICRO_SLICES
from h265_rtp_test import MICRO_SPS
from h265_rtp_test import MICRO_VPS
from h265_rtp_test import BitWriter
from h265_rtp_test import nal
from h265_rtp_test import packetize
import pytest
import rtpcap
import stream_info  # pytest puts this directory on sys.path

# Three pictures (IDR, P, P) of four slices, each slice padded to 3016 bytes (three fragmentation units).
ACCESS_UNITS = [
    [MICRO_VPS, MICRO_SPS, MICRO_PPS, *(s + b"\x55" * 3000 for s in MICRO_SLICES[0])],
    [s + b"\x55" * 3000 for s in MICRO_SLICES[1]],
    [s + b"\x55" * 3000 for s in MICRO_SLICES[2]],
]


def _write_rtpcap(path, access_units, *, first_seq=65500, first_ts=2**32 - 2000) -> list[bytes]:
    packets = []
    for index, units in enumerate(access_units):
        packets += packetize(units, ts=first_ts + 2000 * index, seq=first_seq + len(packets))
    with rtpcap.Writer(path, {"source": "test"}) as writer:
        for send_ns, packet in rtpcap.synthesize_send_times(packets, fps=45, line_rate_bps=1e9):
            writer.write(send_ns, packet)
    return packets


def test_rtpcap_summary(tmp_path):
    packets = _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)
    with open(tmp_path / "x.h265", "wb") as h265_out:
        info = stream_info.analyze_rtpcap(str(tmp_path / "x.rtpcap"), h265_out=h265_out)
    assert info["packets_sha256"] == rtpcap.packets_sha256(packets)
    assert info["meta"] == {"source": "test"}
    rtp = info["rtp"]
    assert (rtp["packets"], rtp["access_units"], rtp["fps_from_timestamps"], rtp["seq_wraps"]) == (
        len(packets),
        3,
        45.0,
        1,
    )
    assert info["depacketizer"] == {
        "packets": {"fu": len(packets) - 3, "single": 3},
        "fu_dropped": 0,
        "access_units_without_marker": 0,
    }
    s = info["stream"]
    assert (s["profile"], s["width"], s["height"], s["chroma_format"], s["bit_depth_luma"]) == (
        "Main",
        256,
        256,
        "4:2:0",
        8,
    )
    assert (s["wpp"], s["tiles"], s["dependent_slice_segments_enabled"], s["ctb_size"], s["sao"]) == (
        True,
        None,
        False,
        32,
        True,
    )
    assert s["slice_segments_per_au"] == {"4": 3}
    assert s["slice_types"] == {"I": 4, "P": 8, "B": 0}
    assert (s["b_slices"], s["reordered_pictures"], s["sps_max_num_reorder_pics"]) == (0, 0, [0])
    assert (s["irap_count"], s["idr_count"], s["first_irap_au"]) == (1, 1, 0)
    assert s["parameter_sets"] == {
        "copies": {"PPS": 1, "SPS": 1, "VPS": 1},
        "distinct": {"PPS": 1, "SPS": 1, "VPS": 1},
        "per_irap_au_vps_sps_pps": {"1/1/1": 1},
        "in_other_aus": 0,
    }
    assert s["fps"] == 45.0
    assert s["es_bytes"] == sum(len(unit) for units in ACCESS_UNITS for unit in units)
    assert s["errors"] == {}
    # The Annex-B output holds the same NAL units, and reads back to the same bitstream summary.
    assert h265_rtp.split_annexb((tmp_path / "x.h265").read_bytes()) == [u for units in ACCESS_UNITS for u in units]
    annexb = stream_info.analyze_annexb(str(tmp_path / "x.h265"), fps=None)
    assert annexb["stream"] == s


def test_lost_packets_show_up(tmp_path):
    packets = _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)
    with rtpcap.Writer(tmp_path / "lossy.rtpcap", {}) as writer:
        for index, packet in enumerate(packets):
            if index != 5:  # the last fragment of the IDR picture's first slice
                writer.write(index, packet)
    info = stream_info.analyze_rtpcap(str(tmp_path / "lossy.rtpcap"))
    assert info["rtp"]["seq_missing"] == 1
    assert info["depacketizer"]["fu_dropped"] == 1
    assert info["stream"]["slice_segments_per_au"] == {"3": 1, "4": 2}


def test_drop_incomplete_access_units(tmp_path):
    """--drop-incomplete-aus leaves out the access units a lost or late packet cut short and keeps their parameter
    sets; without it every access unit is written."""
    packets = _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)  # AU 0: 3 single + 12 FU packets, AUs 1, 2: 12 each
    with rtpcap.Writer(tmp_path / "lossy.rtpcap", {}) as writer:
        for index, packet in enumerate(packets):
            if index != 5:  # the last fragment of the IDR picture's first slice
                writer.write(index, packet)
    with open(tmp_path / "lossy.h265", "wb") as h265_out:
        info = stream_info.analyze_rtpcap(str(tmp_path / "lossy.rtpcap"), h265_out=h265_out, drop_incomplete_aus=True)
    ts = [h265_rtp.parse_rtp_header(packets[i]).timestamp for i in (0, 15, 27)]
    assert info["annexb"] == {
        "drop_incomplete_aus": True,
        "access_units_written": 2,
        "access_units_dropped": 1,
        "dropped": [{"access_unit": 0, "timestamp": ts[0], "why": ["lost or late packet"]}],
    }
    written = h265_rtp.split_annexb((tmp_path / "lossy.h265").read_bytes())
    assert written == [MICRO_VPS, MICRO_SPS, MICRO_PPS, *ACCESS_UNITS[1], *ACCESS_UNITS[2]]
    # Two packets of the second picture swapped: that picture is left out (late packet), the others are whole.
    order = list(range(len(packets)))
    order[17], order[18] = 18, 17
    with rtpcap.Writer(tmp_path / "reordered.rtpcap", {}) as writer:
        for index in order:
            writer.write(index, packets[index])
    with open(tmp_path / "reordered.h265", "wb") as h265_out:
        info = stream_info.analyze_rtpcap(
            str(tmp_path / "reordered.rtpcap"), h265_out=h265_out, drop_incomplete_aus=True
        )
    assert (info["annexb"]["access_units_written"], info["annexb"]["access_units_dropped"]) == (2, 1)
    assert info["annexb"]["dropped"][0]["timestamp"] == ts[1]
    written = h265_rtp.split_annexb((tmp_path / "reordered.h265").read_bytes())
    assert written == [*ACCESS_UNITS[0], *ACCESS_UNITS[2]]
    # Without the flag the summary says so and every access unit goes out.
    with open(tmp_path / "all.h265", "wb") as h265_out:
        info = stream_info.analyze_rtpcap(str(tmp_path / "lossy.rtpcap"), h265_out=h265_out)
    assert (info["annexb"]["drop_incomplete_aus"], info["annexb"]["access_units_written"]) == (False, 3)


# The packets of ACCESS_UNITS as _write_rtpcap sends them: AU 0 is packets 0-14, AU 1 15-26, AU 2 27-38; the last
# packet of each carries the marker bit.
AU_FIRST_PACKETS = (0, 15, 27)
PACKETS = 39
LOST = "lost or late packet"


def _drop_incomplete(tmp_path, name: str, packets: list[bytes]) -> tuple[dict, list[bytes]]:
    """stream_info --drop-incomplete-aus on `packets`, in this order: the summary and the NAL units written."""
    with rtpcap.Writer(tmp_path / f"{name}.rtpcap", {}) as writer:
        for index, packet in enumerate(packets):
            writer.write(index, packet)
    with open(tmp_path / f"{name}.h265", "wb") as h265_out:
        info = stream_info.analyze_rtpcap(str(tmp_path / f"{name}.rtpcap"), h265_out=h265_out, drop_incomplete_aus=True)
    return info, h265_rtp.split_annexb((tmp_path / f"{name}.h265").read_bytes())


@pytest.mark.parametrize(
    ("order", "written", "dropped", "nals"),
    [
        pytest.param(
            [i for i in range(PACKETS) if i != 15],
            1,
            [(0, 0, [LOST]), (1, 1, [LOST])],
            [MICRO_VPS, MICRO_SPS, MICRO_PPS, *ACCESS_UNITS[2]],
            id="gap",
        ),
        pytest.param(
            [i for i in range(PACKETS) if i != 26],
            1,
            [(1, 1, [LOST, "no marker"]), (2, 2, [LOST])],
            ACCESS_UNITS[0],
            id="no-marker",
        ),
        pytest.param(
            [*range(25), 26, 25, *range(27, PACKETS)],
            2,
            [(1, 1, [LOST]), (2, 1, [LOST, "no marker", "no slice"])],
            [*ACCESS_UNITS[0], *ACCESS_UNITS[2]],
            id="no-slice",
        ),
    ],
)
def test_drop_incomplete_access_units_at_their_boundaries(tmp_path, order, written, dropped, nals):
    """--drop-incomplete-aus with a packet lost or late at the boundary of two access units, as in the real fixture
    (B70 client project logs/pB2/pic_20261103_203057/stream_info_real_200s.txt; B2 review J10):

    - gap: AU 1's first packet is lost. The gap shows at AU 1's second packet, right after AU 0's last one, so AU 0 is
      left out as well, although its marker had come: the unit before a gap may be the one cut short.
    - no-marker: AU 1's last packet, the one with the marker bit, is lost. AU 1 ends only when AU 2's timestamp
      starts, and AU 2, on the other side of the gap, is left out too.
    - no-slice: AU 1's last two packets arrive swapped. Its marker ends AU 1 (whose last slice is lost), and the late
      packet makes a unit of its own, with AU 1's timestamp, no marker and no slice.

    `dropped` lists (index of the access unit, the AU whose timestamp it has, why); the parameter sets of the access
    units left out are still written."""
    packets = _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)
    timestamps = [h265_rtp.parse_rtp_header(packets[i]).timestamp for i in AU_FIRST_PACKETS]
    info, written_nals = _drop_incomplete(tmp_path, "lossy", [packets[i] for i in order])
    assert info["annexb"] == {
        "drop_incomplete_aus": True,
        "access_units_written": written,
        "access_units_dropped": len(dropped),
        "dropped": [{"access_unit": index, "timestamp": timestamps[au], "why": why} for index, au, why in dropped],
    }
    assert written_nals == nals


def test_drop_incomplete_access_units_without_a_marker_or_a_slice(tmp_path):
    """With no packet lost or late, an access unit is left out all the same when its last packet lacks the marker bit
    (it ends only when the next timestamp starts), or when it holds no slice (parameter sets only: still written)."""
    packets = _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)
    timestamps = [h265_rtp.parse_rtp_header(packets[i]).timestamp for i in AU_FIRST_PACKETS]
    unmarked = list(packets)
    unmarked[26] = packets[26][:1] + bytes((packets[26][1] & 0x7F,)) + packets[26][2:]  # AU 1's last packet
    info, nals = _drop_incomplete(tmp_path, "unmarked", unmarked)
    assert info["annexb"]["dropped"] == [{"access_unit": 1, "timestamp": timestamps[1], "why": ["no marker"]}]
    assert info["annexb"]["access_units_written"] == 2
    assert nals == [*ACCESS_UNITS[0], *ACCESS_UNITS[2]]
    # AU 0's parameter sets sent as an access unit of their own, ahead of the IDR picture.
    packets = _write_rtpcap(tmp_path / "y.rtpcap", [ACCESS_UNITS[0][:3], ACCESS_UNITS[0][3:], *ACCESS_UNITS[1:]])
    info, nals = _drop_incomplete(tmp_path, "parameter_sets_alone", packets)
    first = h265_rtp.parse_rtp_header(packets[0]).timestamp
    assert info["annexb"]["dropped"] == [{"access_unit": 0, "timestamp": first, "why": ["no slice"]}]
    assert info["annexb"]["access_units_written"] == 3
    assert nals == [unit for units in ACCESS_UNITS for unit in units]


def test_reordered_pictures_and_b_slices():
    sps = h265_rtp.parse_sps(MICRO_SPS)

    def picture(slice_type: int, poc_lsb: int) -> list[bytes]:
        header = BitWriter().u(1, 1).ue(0).ue(slice_type).u(sps.log2_max_pic_order_cnt_lsb, poc_lsb)
        return [nal(1, header.rbsp() + b"\x55" * 10)]

    stats = stream_info.BitstreamStats()
    stats.add_access_unit(ACCESS_UNITS[0])
    for slice_type, poc in ((h265_rtp.SLICE_P, 2), (h265_rtp.SLICE_B, 1), (h265_rtp.SLICE_P, 4), (h265_rtp.SLICE_B, 3)):
        stats.add_access_unit(picture(slice_type, poc))
    stats.add_access_unit(ACCESS_UNITS[0])  # a new IDR: POC 0 again is not reordering
    stats.add_access_unit(picture(h265_rtp.SLICE_P, 1))
    summary = stats.summary(fps=45.0)
    assert summary["slice_types"] == {"I": 8, "P": 3, "B": 2}
    assert summary["reordered_pictures"] == 2
    assert summary["irap_interval_frames"] == {"5": 1}
    assert summary["irap_interval_s"] == {"0.1111": 1}
    assert summary["parameter_sets"]["per_irap_au_vps_sps_pps"] == {"1/1/1": 2}


def test_command_line(tmp_path, capsys):
    _write_rtpcap(tmp_path / "x.rtpcap", ACCESS_UNITS)
    stream_info.main(
        [str(tmp_path / "x.rtpcap"), "--json", str(tmp_path / "x.json"), "--write-h265", str(tmp_path / "x.h265")]
    )
    printed = capsys.readouterr().out
    assert "WPP True" in printed
    assert "slice types {'I': 4, 'P': 8, 'B': 0}" in printed
    info = json.loads((tmp_path / "x.json").read_text())
    assert (info["format"], info["stream"]["access_units"]) == ("rtpcap", 3)
    stream_info.main([str(tmp_path / "x.h265"), "--json", "-"])
    assert json.loads(capsys.readouterr().out)["stream"]["slice_segments_per_au"] == {"4": 3}
    stream_info.main([str(tmp_path / "x.rtpcap"), "--write-h265", str(tmp_path / "y.h265"), "--drop-incomplete-aus"])
    assert "3 access units written, 0 left out" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        stream_info.main([str(tmp_path / "x.rtpcap"), "--drop-incomplete-aus"])
