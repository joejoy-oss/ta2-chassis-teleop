"""Tests for h265_rtp.py: RTP headers, RFC 7798 depacketizing, access units, Annex-B, bit reading, parameter sets and
slice headers. The parameter sets and slice header bytes are real x265 3.5 output (logs/p1/12, 13 of the B70 client
project): the synthetic fixtures' 1280x2720 stream, and a 256x256 WPP stream with 4 slices per picture."""

from __future__ import annotations

import dataclasses
import io
import struct

import h265_rtp  # pytest puts this directory on sys.path
import pytest

# The synthetic fixtures (make_synthetic.sh): 1280x2720, CTU 32, no WPP.
FIXTURE_VPS = bytes.fromhex("40010c01ffff016000000300900000030000030096ba0240")
FIXTURE_SPS = bytes.fromhex(
    "420101016000000300900000030000030096a00280800aa165bab932bffc00040005a8080808200000030020000005a1"
)
FIXTURE_PPS = bytes.fromhex("4401c173c089")
# First 16 bytes of its first three slices (logs/p1/15): IDR (RTP timestamp 4294000000), P (+1999), P (+3999).
FIXTURE_SLICES = [
    bytes.fromhex("2801af11802ebae7323fff7bae0e8ea2"),
    bytes.fromhex("0201d0097e205b63f8f6bd4e3d70e6fd"),
    bytes.fromhex("0201d011ff881e40c215c03b1220ec4e"),
]
# The same encoder settings at 256x256 with WPP and slices=4 (x265 allows several slices only with WPP).
MICRO_VPS = bytes.fromhex("40010c01ffff01600000030090000003000003003fba0240")
MICRO_SPS = bytes.fromhex(
    "42010101600000030090000003000003003fa00808040596eae4cafff000100016a0c0c0c080000003008000001684"
)
MICRO_PPS = bytes.fromhex("4401c173c109")
# First 16 bytes of the slice NAL units of its three pictures (IDR, P, P), four slices each.
MICRO_SLICES = [
    [
        bytes.fromhex("2801af0ca11cb04a131a6e3faff5325c"),
        bytes.fromhex("2801283c328472c04a131a6e3faff532"),
        bytes.fromhex("2801303c328472c04a131a6e3faff532"),
        bytes.fromhex("2801383c3285f998490387fffcd587d0"),
    ],
    [
        bytes.fromhex("0201d0097e206d1b28a916eb5cbe661d"),
        bytes.fromhex("0201504025f881b46ca0a916eb5cbe66"),
        bytes.fromhex("0201604025f881b46ca0a916eb5cbe66"),
        bytes.fromhex("0201704025f881b43204e0a7f370d6b9"),
    ],
    [
        bytes.fromhex("0201d011ff8808d1ac805375b7c97cbb"),
        bytes.fromhex("0201504047fe202346b25375b7c97cbb"),
        bytes.fromhex("0201604047fe202346b25375b7c97cbb"),
        bytes.fromhex("0201704047fe202342ffdc2569724011"),
    ],
]


class BitWriter:
    """Builds RBSPs bit by bit for hand-made parameter sets and slice headers (also used by stream_info_test.py)."""

    def __init__(self):
        self.bits: list[int] = []

    def u(self, n: int, value: int) -> BitWriter:
        self.bits.extend((value >> (n - 1 - i)) & 1 for i in range(n))
        return self

    def ue(self, value: int) -> BitWriter:
        code = value + 1
        return self.u(code.bit_length() - 1, 0).u(code.bit_length(), code)

    def se(self, value: int) -> BitWriter:
        return self.ue(2 * value - 1 if value > 0 else -2 * value)

    def rbsp(self) -> bytes:
        """The bits with rbsp_trailing_bits (a one, then zeros to the byte boundary)."""
        bits = [*self.bits, 1]
        bits += [0] * (-len(bits) % 8)
        return bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))


def escape(rbsp: bytes) -> bytes:
    """Insert emulation prevention bytes (the inverse of h265_rtp.unescape_rbsp)."""
    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def nal(nal_type: int, rbsp: bytes, *, layer_id: int = 0, tid: int = 0) -> bytes:
    header = bytes(((nal_type << 1) | (layer_id >> 5), ((layer_id & 0x1F) << 3) | (tid + 1)))
    return header + escape(rbsp)


def rtp_packet(payload: bytes, *, seq: int, ts: int, marker: bool = False, pt: int = 96, ssrc: int = 0x1234) -> bytes:
    return struct.pack("!BBHII", 0x80, (0x80 if marker else 0) | pt, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc) + payload


def packetize(nals: list[bytes], *, ts: int, seq: int, mtu: int = 1400) -> list[bytes]:
    """RFC 7798 packets of one access unit (single NAL unit packets, FUs for large units, marker on the last), the
    way rtph265pay sends the fixtures (also used by stream_info_test.py)."""
    payloads = []
    for unit in nals:
        if len(unit) + 12 <= mtu:
            payloads.append(unit)
            continue
        header = bytes(((unit[0] & 0x81) | (h265_rtp.FU << 1), unit[1]))
        body = unit[2:]
        size = mtu - 12 - 3
        pieces = [body[i : i + size] for i in range(0, len(body), size)]
        for index, piece in enumerate(pieces):
            flags = (0x80 if index == 0 else 0) | (0x40 if index == len(pieces) - 1 else 0)
            payloads.append(header + bytes((flags | h265_rtp.nal_type(unit),)) + piece)
    return [
        rtp_packet(payload, seq=seq + i, ts=ts, marker=i == len(payloads) - 1) for i, payload in enumerate(payloads)
    ]


def _depacketize(packets: list[bytes], **kwargs) -> tuple[h265_rtp.Depacketizer, list[bytes]]:
    depacketizer = h265_rtp.Depacketizer(**kwargs)
    nals = []
    for packet in packets:
        header = h265_rtp.parse_rtp_header(packet)
        nals += depacketizer.feed(header, packet[header.payload_offset : header.payload_end])
    return depacketizer, nals


# --- RTP headers ------------------------------------------------------------------------------------------------------


def test_rtp_header_fixed_fields():
    header = h265_rtp.parse_rtp_header(rtp_packet(b"\x02\x01xyz", seq=65535, ts=0xFFFFFFFF, marker=True, pt=96))
    assert (header.marker, header.payload_type, header.sequence, header.timestamp, header.ssrc) == (
        True,
        96,
        65535,
        0xFFFFFFFF,
        0x1234,
    )
    assert header.csrcs == ()
    assert header.extension_profile is None
    assert (header.payload_offset, header.payload_end, header.padding) == (12, 17, 0)


def test_rtp_header_csrcs_extension_and_padding():
    csrcs = struct.pack("!II", 7, 8)
    extension = struct.pack("!HH", 0xBEDE, 2) + b"\x10\xaa\x00\x00" + b"\x00" * 4
    payload = b"\x02\x01payload"
    padding = b"\x00\x00\x03"
    first = 0x80 | 0x20 | 0x10 | 2  # version 2, padding, extension, 2 CSRCs
    packet = struct.pack("!BBHII", first, 96, 5, 6, 9) + csrcs + extension + payload + padding
    header = h265_rtp.parse_rtp_header(packet)
    assert header.csrcs == (7, 8)
    assert (header.extension_profile, header.extension_length, header.padding) == (0xBEDE, 8, 3)
    assert packet[header.payload_offset : header.payload_end] == payload
    # From a prefix of the packet (as rtp_capture keeps it) the padding is not looked at.
    prefix_header = h265_rtp.parse_rtp_header(packet[:40], prefix=True)
    assert prefix_header.payload_offset == header.payload_offset
    assert prefix_header.padding == 0


@pytest.mark.parametrize(
    ("packet", "message"),
    [
        (b"\x80\x60\x00", "shorter"),
        (b"\x40" + b"\x00" * 11, "version"),
        (b"\x83" + b"\x00" * 11 + b"\x00" * 8, "CSRC"),
        (b"\x90" + b"\x00" * 11 + b"\xbe\xde\x00\x05", "extension"),
        (b"\xa0" + b"\x00" * 11 + b"\x02\x01\x00", "padding"),
        (b"\xa0" + b"\x00" * 11 + b"\x02\x01\x09", "padding"),
    ],
)
def test_rtp_header_errors(packet, message):
    with pytest.raises(h265_rtp.RtpError, match=message):
        h265_rtp.parse_rtp_header(packet)


# --- RFC 7798 payloads ------------------------------------------------------------------------------------------------


def test_single_nal_unit_packets_including_header_only_units():
    eos = bytes((36 << 1, 1))  # an end-of-sequence NAL unit is only its 2-byte header
    depacketizer, nals = _depacketize([rtp_packet(FIXTURE_VPS, seq=1, ts=0), rtp_packet(eos, seq=2, ts=0, marker=True)])
    assert nals == [FIXTURE_VPS, eos]
    assert depacketizer.packets == {"single": 2}


def test_aggregation_packet():
    units = [FIXTURE_VPS, FIXTURE_SPS, FIXTURE_PPS]
    payload = bytes((h265_rtp.AP << 1, 1)) + b"".join(struct.pack("!H", len(u)) + u for u in units)
    depacketizer, nals = _depacketize([rtp_packet(payload, seq=1, ts=0)])
    assert nals == units
    assert depacketizer.packets == {"ap": 1}


def test_aggregation_packet_with_don_fields():
    units = [FIXTURE_VPS, FIXTURE_PPS]
    payload = (
        bytes((h265_rtp.AP << 1, 1))
        + b"\x00\x07"  # DONL of the first unit
        + struct.pack("!H", len(units[0]))
        + units[0]
        + b"\x00"  # DOND of the second
        + struct.pack("!H", len(units[1]))
        + units[1]
    )
    _depacketizer, nals = _depacketize([rtp_packet(payload, seq=1, ts=0)], donl=True)
    assert nals == units


def test_truncated_aggregation_packet_is_invalid():
    payload = bytes((h265_rtp.AP << 1, 1)) + struct.pack("!H", 50) + FIXTURE_PPS
    depacketizer, nals = _depacketize([rtp_packet(payload, seq=1, ts=0)])
    assert nals == []
    assert depacketizer.packets["invalid"] == 1


def test_fragmentation_units_rebuild_the_nal_unit_header():
    unit = nal(h265_rtp.IDR_N_LP, bytes(range(256)) * 12, layer_id=0, tid=2)
    packets = packetize([unit], ts=0, seq=65534, mtu=500)  # sequence numbers wrap inside the unit
    assert len(packets) > 3
    depacketizer, nals = _depacketize(packets)
    assert nals == [unit]
    assert h265_rtp.nal_temporal_id(nals[0]) == 2
    assert depacketizer.packets == {"fu": len(packets)}
    assert depacketizer.fu_dropped == 0


def test_lost_fragment_drops_the_unit_once():
    units = [nal(1, b"\x55" * 3000), nal(1, b"\xaa" * 3000)]  # 4 fragments each
    packets = packetize(units[:1], ts=0, seq=10, mtu=1000) + packetize(units[1:], ts=2000, seq=14, mtu=1000)
    assert len(packets) == 8
    lost_middle = packets[:1] + packets[2:]  # the second fragment of the first unit is missing
    depacketizer, nals = _depacketize(lost_middle)
    assert nals == units[1:]
    assert depacketizer.fu_dropped == 1


def test_fragments_without_their_start_are_dropped_once():
    unit = nal(1, b"\x55" * 3000)
    packets = packetize([unit, unit], ts=0, seq=0, mtu=1000)  # 4 fragments each
    depacketizer, nals = _depacketize(packets[1:])  # the first unit lost its start
    assert nals == [unit]
    assert depacketizer.fu_dropped == 1


def test_new_start_before_end_and_unfinished_unit_count_as_dropped():
    unit = nal(1, b"\x55" * 3000)
    first = packetize([unit], ts=0, seq=0, mtu=1000)
    second = packetize([unit], ts=0, seq=3, mtu=1000)
    depacketizer, nals = _depacketize(first[:2] + second + second[:1])
    depacketizer.flush()
    assert nals == [unit]
    assert depacketizer.fu_dropped == 2


def test_paci_unknown_and_invalid_payloads():
    depacketizer, nals = _depacketize(
        [
            rtp_packet(bytes((h265_rtp.PACI << 1, 1, 0, 0)), seq=1, ts=0),
            rtp_packet(bytes((60 << 1, 1, 0)), seq=2, ts=0),
            rtp_packet(bytes((0x80 | 2, 1, 0)), seq=3, ts=0),  # forbidden bit set
            rtp_packet(bytes((h265_rtp.FU << 1, 1, 0xC0 | 1, 0)), seq=4, ts=0),  # FU with start and end
            rtp_packet(b"\x02", seq=5, ts=0),  # no payload header
        ]
    )
    assert nals == []
    assert depacketizer.packets == {"paci": 1, "unknown": 1, "invalid": 3}


def test_rtp_access_units_group_by_timestamp_and_marker():
    units = h265_rtp.RtpAccessUnits()
    done = []
    for seq, (ts, marker) in enumerate([(0, False), (0, True), (2000, False), (4000, True)]):
        header = h265_rtp.parse_rtp_header(rtp_packet(b"\x02\x01", seq=seq, ts=ts, marker=marker))
        done += units.add(header, [bytes((2, 1, seq))], 100 * seq)
    assert units.flush() is None
    assert [(au.timestamp, au.packets, au.marker, au.first_time_ns, au.last_time_ns) for au in done] == [
        (0, 2, True, 0, 100),
        (2000, 1, False, 200, 200),  # its marker packet was lost: the next timestamp ends it
        (4000, 1, True, 300, 300),
    ]
    assert done[0].nals == [b"\x02\x01\x00", b"\x02\x01\x01"]


# --- Annex-B ----------------------------------------------------------------------------------------------------------


def test_split_annexb_with_both_start_codes_and_trailing_zeros():
    stream = b"\x00\x00\x00\x01" + FIXTURE_VPS + b"\x00\x00" + b"\x00\x00\x01" + FIXTURE_SPS + b"\x00\x00\x01\x26\x01"
    assert h265_rtp.split_annexb(stream) == [FIXTURE_VPS, FIXTURE_SPS, b"\x26\x01"]
    assert h265_rtp.split_annexb(h265_rtp.to_annexb([FIXTURE_VPS, FIXTURE_PPS])) == [FIXTURE_VPS, FIXTURE_PPS]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 64, 1 << 20])
def test_iter_annexb_matches_split_for_any_chunking(chunk_size):
    units = [FIXTURE_VPS, FIXTURE_SPS, FIXTURE_PPS, *MICRO_SLICES[0], nal(1, bytes(500) + b"\x01")]
    stream = b"\xff\xfe" + h265_rtp.to_annexb(units) + b"\x00\x00"
    assert list(h265_rtp.iter_annexb(io.BytesIO(stream), chunk_size=chunk_size)) == h265_rtp.split_annexb(stream)
    assert h265_rtp.split_annexb(stream) == units


def test_annexb_access_units():
    units = h265_rtp.AnnexBAccessUnits()
    sei = bytes((h265_rtp.PREFIX_SEI_NUT << 1, 1, 5, 0x80))
    sequence = [FIXTURE_VPS, FIXTURE_SPS, FIXTURE_PPS, *MICRO_SLICES[0], sei, *MICRO_SLICES[1], *MICRO_SLICES[2]]
    done = [au for au in (units.add(unit) for unit in sequence) if au is not None]
    done.append(units.flush())
    assert done == [[FIXTURE_VPS, FIXTURE_SPS, FIXTURE_PPS, *MICRO_SLICES[0]], [sei, *MICRO_SLICES[1]], MICRO_SLICES[2]]
    assert units.flush() is None


# --- Bit reading ------------------------------------------------------------------------------------------------------


def test_unescape_rbsp():
    assert h265_rtp.unescape_rbsp(b"\x00\x00\x03\x01") == b"\x00\x00\x01"
    assert h265_rtp.unescape_rbsp(b"\x00\x00\x03\x00\x00\x03") == b"\x00\x00\x00\x00"
    assert h265_rtp.unescape_rbsp(b"\x00\x00\x03\x03\x00\x03") == b"\x00\x00\x03\x00\x03"
    rbsp = bytes([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 7])
    assert h265_rtp.unescape_rbsp(escape(rbsp)) == rbsp


def test_exp_golomb_codes():
    # ue(v): 1 -> 0, 010 -> 1, 011 -> 2, 00100 -> 3, 00111 -> 6; se(v) maps 1, 2, 3, 4 to 1, -1, 2, -2.
    reader = h265_rtp.BitReader(bytes((0b10100110, 0b01000011, 0b10000000)))
    assert [reader.ue(), reader.ue(), reader.ue(), reader.ue(), reader.ue()] == [0, 1, 2, 3, 6]
    writer = BitWriter()
    for value in (0, 1, -1, 2, -2, 1000, -1000):
        writer.se(value)
    for value in (0, 31, 32, 1 << 20):
        writer.ue(value)
    writer.u(13, 0x1ABC)
    reader = h265_rtp.BitReader(writer.rbsp())
    assert [reader.se() for _ in range(7)] == [0, 1, -1, 2, -2, 1000, -1000]
    assert [reader.ue() for _ in range(4)] == [0, 31, 32, 1 << 20]
    assert reader.u(13) == 0x1ABC
    assert reader.flag()  # the rbsp stop bit
    assert reader.bits_left < 8


def test_bit_reader_errors():
    reader = h265_rtp.BitReader(b"\xff")
    reader.u(6)
    with pytest.raises(h265_rtp.BitstreamError):
        reader.u(3)
    with pytest.raises(h265_rtp.BitstreamError, match="longer than 32"):
        h265_rtp.BitReader(bytes(5) + b"\xff").ue()
    with pytest.raises(h265_rtp.BitstreamError):
        h265_rtp.BitReader(b"\x00").ue()


def test_nal_unit_header_fields():
    unit = nal(h265_rtp.CRA_NUT, b"\x80", layer_id=33, tid=4)
    assert (h265_rtp.nal_type(unit), h265_rtp.nal_layer_id(unit), h265_rtp.nal_temporal_id(unit)) == (21, 33, 4)
    assert [h265_rtp.is_irap(kind) for kind in (15, 16, 21, 23, 24)] == [False, True, True, True, False]
    assert [h265_rtp.is_idr(kind) for kind in (18, 19, 20, 21)] == [False, True, True, False]
    assert h265_rtp.nal_type_name(20) == "IDR_N_LP"
    assert h265_rtp.nal_type_name(45) == "type45"


# --- Parameter sets ---------------------------------------------------------------------------------------------------


def test_fixture_vps():
    vps = h265_rtp.parse_vps(FIXTURE_VPS)
    assert (vps.vps_id, vps.max_layers, vps.max_sub_layers) == (0, 1, 1)
    assert vps.profile_tier_level.profile_name == "Main"
    assert vps.profile_tier_level.level == 5.0
    assert vps.max_num_reorder_pics == [0]


def test_fixture_sps():
    sps = h265_rtp.parse_sps(FIXTURE_SPS)
    assert (sps.width, sps.height, sps.cropped_size) == (1280, 2720, (1280, 2720))
    assert (sps.chroma_format_idc, sps.bit_depth_luma, sps.bit_depth_chroma) == (1, 8, 8)
    assert (sps.ctb_size, sps.log2_min_coding_block_size, sps.width_in_ctbs, sps.height_in_ctbs) == (32, 3, 40, 85)
    assert sps.slice_segment_address_bits == 12  # 3400 CTBs
    assert sps.sample_adaptive_offset_enabled
    assert (sps.amp_enabled, sps.pcm_enabled, sps.scaling_list_enabled) == (False, False, False)
    assert sps.max_num_reorder_pics == [0]
    assert sps.profile_tier_level.profile_idc == 1
    assert sps.vui is not None
    assert sps.vui.fps == 45.0
    assert (sps.vui.video_full_range, sps.vui.matrix_coefficients) == (False, 1)  # limited range, BT.709


def test_fixture_and_micro_pps():
    pps = h265_rtp.parse_pps(FIXTURE_PPS)
    assert (pps.entropy_coding_sync_enabled, pps.tiles_enabled) == (False, False)
    assert (pps.dependent_slice_segments_enabled, pps.num_extra_slice_header_bits) == (False, 0)
    assert (pps.init_qp, pps.sign_data_hiding_enabled, pps.cu_qp_delta_enabled) == (26, True, True)
    assert (pps.loop_filter_across_slices_enabled, pps.deblocking_filter_disabled, pps.weighted_pred) == (
        True,
        False,
        False,
    )
    micro = h265_rtp.parse_pps(MICRO_PPS)
    assert micro.entropy_coding_sync_enabled
    assert not micro.loop_filter_across_slices_enabled  # x265 turns it off with several slices


def _rich_sps_rbsp() -> bytes:
    """An SPS with what x265 does not write: conformance window, 10 bit, scaling lists, PCM, two short-term RPS (the
    second predicted from the first), long-term pictures and a full VUI start."""
    w = BitWriter()
    w.u(4, 0).u(3, 0).u(1, 1)  # vps id, max_sub_layers_minus1, temporal_id_nesting
    w.u(2, 0).u(1, 0).u(5, 2).u(32, 0x20000000).u(1, 1).u(1, 0).u(1, 0).u(1, 1).u(43, 0).u(1, 0).u(8, 123)
    w.ue(0).ue(1).ue(1920).ue(1080)  # sps id, 4:2:0, size
    w.u(1, 1).ue(0).ue(0).ue(0).ue(4)  # conformance window: 4 chroma rows (8 luma rows) off the bottom
    w.ue(2).ue(2).ue(4)  # 10-bit luma and chroma, log2_max_pic_order_cnt_lsb 8
    w.u(1, 1).ue(4).ue(2).ue(0)  # sub-layer ordering: dec_pic_buffering_minus1 4, num_reorder 2
    w.ue(0).ue(3).ue(0).ue(3).ue(1).ue(1)  # CB 8..64, TB 4..32, transform hierarchy depths
    w.u(1, 1).u(1, 1)  # scaling_list_enabled, data present
    for size_id in range(4):
        for _ in range(0, 6, 3 if size_id == 3 else 1):
            w.u(1, 1)  # scaling_list_pred_mode_flag: explicit coefficients
            if size_id > 1:
                w.se(-3)
            for _ in range(min(64, 1 << (4 + (size_id << 1)))):
                w.se(1)
    w.u(1, 1).u(1, 1)  # amp, sao
    w.u(1, 1).u(4, 7).u(4, 7).ue(0).ue(1).u(1, 0)  # pcm
    w.ue(2)  # num_short_term_ref_pic_sets
    w.ue(2).ue(1)  # set 0: 2 negative, 1 positive pictures
    for _ in range(3):
        w.ue(0).u(1, 1)
    w.u(1, 1).u(1, 0).ue(0)  # set 1: predicted from set 0, delta_rps_sign, abs_delta_rps_minus1
    for used, use_delta in ((1, None), (0, 1), (0, 0), (1, None)):  # NumDeltaPocs[0] + 1 = 4 entries
        w.u(1, used)
        if use_delta is not None:
            w.u(1, use_delta)
    w.u(1, 1).ue(1).u(8, 77).u(1, 1)  # long-term pictures: one in the SPS
    w.u(1, 1).u(1, 0)  # temporal mvp, no strong intra smoothing
    w.u(1, 1)  # vui_parameters_present_flag
    w.u(1, 1).u(8, 255).u(16, 4).u(16, 3)  # extended SAR
    w.u(1, 0)  # overscan
    w.u(1, 1).u(3, 5).u(1, 1).u(1, 1).u(8, 9).u(8, 16).u(8, 9)  # full range, BT.2020 PQ
    w.u(1, 1).ue(2).ue(2)  # chroma location
    w.u(1, 0).u(1, 0).u(1, 0)
    w.u(1, 1).ue(1).ue(1).ue(1).ue(1)  # default display window
    w.u(1, 1).u(32, 1001).u(32, 60000)  # timing
    w.u(1, 0).u(1, 0)  # poc proportional, no HRD; the rest is not parsed
    return w.rbsp()


def test_rich_sps():
    sps = h265_rtp.parse_sps(nal(h265_rtp.SPS_NUT, _rich_sps_rbsp()))
    assert sps.profile_tier_level.profile_name == "Main 10"
    assert sps.profile_tier_level.level == pytest.approx(4.1)
    assert (sps.width, sps.height, sps.cropped_size) == (1920, 1080, (1920, 1072))
    assert (sps.bit_depth_luma, sps.log2_max_pic_order_cnt_lsb, sps.ctb_size) == (10, 8, 64)
    assert (sps.max_dec_pic_buffering_minus1, sps.max_num_reorder_pics) == ([4], [2])
    assert (sps.scaling_list_enabled, sps.pcm_enabled, sps.amp_enabled) == (True, True, True)
    assert (sps.num_short_term_ref_pic_sets, sps.long_term_ref_pics_present) == (2, True)
    assert (sps.temporal_mvp_enabled, sps.strong_intra_smoothing_enabled) == (True, False)
    vui = sps.vui
    assert (vui.video_full_range, vui.colour_primaries, vui.transfer_characteristics, vui.matrix_coefficients) == (
        True,
        9,
        16,
        9,
    )
    assert vui.fps == pytest.approx(59.94, abs=0.01)
    assert dataclasses.replace(sps, conformance_window=(2, 2, 0, 0)).cropped_size == (1912, 1080)


def test_truncated_sps_raises():
    with pytest.raises(h265_rtp.BitstreamError, match="SPS"):
        h265_rtp.parse_sps(FIXTURE_SPS[:12])


# --- Slice headers ----------------------------------------------------------------------------------------------------


def test_fixture_slice_headers():
    parameter_sets = ({0: h265_rtp.parse_pps(FIXTURE_PPS)}, {0: h265_rtp.parse_sps(FIXTURE_SPS)})
    headers = [h265_rtp.parse_slice_header(s, *parameter_sets) for s in FIXTURE_SLICES]
    assert headers == [
        (True, 0, False, 0, h265_rtp.SLICE_I, None),
        (True, 0, False, 0, h265_rtp.SLICE_P, 1),
        (True, 0, False, 0, h265_rtp.SLICE_P, 2),
    ]


def test_micro_slice_headers():
    sps = h265_rtp.parse_sps(MICRO_SPS)
    pps = h265_rtp.parse_pps(MICRO_PPS)
    assert sps.slice_segment_address_bits == 6  # 8 x 8 CTBs
    headers = [[h265_rtp.parse_slice_header(s, {0: pps}, {0: sps}) for s in picture] for picture in MICRO_SLICES]
    expected = [(h265_rtp.SLICE_I, None), (h265_rtp.SLICE_P, 1), (h265_rtp.SLICE_P, 2)]  # IDR POC 0, then 1, 2
    for picture, (slice_type, poc_lsb) in zip(headers, expected, strict=True):
        assert [h.first_slice_segment_in_pic for h in picture] == [True, False, False, False]
        assert [h.segment_address for h in picture] == [0, 16, 32, 48]
        assert {h.slice_type for h in picture} == {slice_type}
        assert {h.pic_order_cnt_lsb for h in picture} == {poc_lsb}
        assert not any(h.dependent for h in picture)


def _dependent_slices_pps() -> bytes:
    """PPS 0 with dependent slice segments, pic_output_flag present, 2 extra slice header bits, 4x2 tiles."""
    w = BitWriter().ue(0).ue(0).u(1, 1).u(1, 1).u(3, 2).u(1, 0).u(1, 0).ue(0).ue(0).se(-4)
    w.u(1, 0).u(1, 0).u(1, 0).se(0).se(0).u(1, 0).u(1, 0).u(1, 0).u(1, 0)
    w.u(1, 1).u(1, 0)  # tiles, no WPP
    w.ue(3).ue(1).u(1, 0).ue(1).ue(1).ue(1).ue(3).u(1, 1)  # 4 x 2 tiles, explicit sizes
    w.u(1, 1).u(1, 1).u(1, 1).u(1, 1)  # across slices; deblocking control: override enabled, disabled in PPS
    w.u(1, 0).u(1, 0).ue(0).u(1, 0).u(1, 0)
    return nal(h265_rtp.PPS_NUT, w.rbsp())


def test_tiles_and_dependent_slice_segments():
    sps = h265_rtp.parse_sps(MICRO_SPS)
    pps = h265_rtp.parse_pps(_dependent_slices_pps())
    assert (pps.dependent_slice_segments_enabled, pps.output_flag_present, pps.num_extra_slice_header_bits) == (
        True,
        True,
        2,
    )
    assert (pps.tiles_enabled, pps.num_tile_columns, pps.num_tile_rows, pps.uniform_tile_spacing) == (True, 4, 2, False)
    assert (pps.init_qp, pps.deblocking_filter_override_enabled, pps.deblocking_filter_disabled) == (22, True, True)
    bits = sps.log2_max_pic_order_cnt_lsb
    dependent = nal(1, BitWriter().u(1, 0).ue(0).u(1, 1).u(6, 40).rbsp())
    independent = nal(1, BitWriter().u(1, 0).ue(0).u(1, 0).u(6, 8).u(2, 3).ue(0).u(1, 1).u(bits, 5).rbsp())
    parameter_sets = ({0: pps}, {0: sps})
    assert h265_rtp.parse_slice_header(dependent, *parameter_sets) == (False, 0, True, 40, None, None)
    header = h265_rtp.parse_slice_header(independent, *parameter_sets)
    assert header == (False, 0, False, 8, h265_rtp.SLICE_B, 5)


def test_slice_header_needs_its_parameter_sets():
    with pytest.raises(h265_rtp.BitstreamError, match="PPS 0"):
        h265_rtp.parse_slice_header(MICRO_SLICES[1][0], {}, {})
    pps = h265_rtp.parse_pps(MICRO_PPS)
    with pytest.raises(h265_rtp.BitstreamError, match="SPS 0"):
        h265_rtp.parse_slice_header(MICRO_SLICES[1][0], {0: pps}, {})
