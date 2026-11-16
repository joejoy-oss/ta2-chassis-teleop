"""RTP and H.265 parsing for the TeleAvatar V2 camera stream: RTP headers, RFC 7798 depacketizing into NAL units,
access units, Annex-B byte streams, and the parameter-set and slice-header fields that decide how a decoder can run
(profile, size, bit depth, WPP, tiles, slices per picture, slice types, picture reordering).

Pure Python (no gi, no numpy), used by rtp_capture.py, stream_info.py and their tests.

Limits: RTP header extensions are skipped, not interpreted. PACI packets (RFC 7798 type 50) are recognized and
counted but not unpacked. DONL/DOND fields only exist when the SDP sets sprop-max-don-diff > 0; they are assumed
absent unless donl=True. VPS, SPS and PPS are parsed up to the fields listed in their classes, slice segment headers
up to slice_pic_order_cnt_lsb.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator
import dataclasses
import struct
from typing import BinaryIO, NamedTuple

RTP_VERSION = 2
RTP_HEADER_SIZE = 12
# RFC 7798: H.265 over RTP always uses a 90 kHz timestamp clock.
RTP_CLOCK_RATE = 90_000

# RFC 7798 payload structures; they reuse the NAL unit type field of the payload header (48-63 are unspecified in
# H.265 itself).
AP = 48  # aggregation packet: several NAL units, each with a 16-bit size
FU = 49  # fragmentation unit: one piece of a NAL unit
PACI = 50  # payload content information: not supported here

# NAL unit types (H.265 table 7-1) that the tools refer to.
BLA_W_LP = 16
IDR_W_RADL = 19
IDR_N_LP = 20
CRA_NUT = 21
RSV_IRAP_VCL23 = 23
VPS_NUT = 32
SPS_NUT = 33
PPS_NUT = 34
AUD_NUT = 35
PREFIX_SEI_NUT = 39
SUFFIX_SEI_NUT = 40

NAL_TYPE_NAMES = {
    0: "TRAIL_N",
    1: "TRAIL_R",
    2: "TSA_N",
    3: "TSA_R",
    4: "STSA_N",
    5: "STSA_R",
    6: "RADL_N",
    7: "RADL_R",
    8: "RASL_N",
    9: "RASL_R",
    16: "BLA_W_LP",
    17: "BLA_W_RADL",
    18: "BLA_N_LP",
    19: "IDR_W_RADL",
    20: "IDR_N_LP",
    21: "CRA_NUT",
    32: "VPS",
    33: "SPS",
    34: "PPS",
    35: "AUD",
    36: "EOS",
    37: "EOB",
    38: "FD",
    39: "PREFIX_SEI",
    40: "SUFFIX_SEI",
    AP: "AP",
    FU: "FU",
    PACI: "PACI",
}

SLICE_B = 0
SLICE_P = 1
SLICE_I = 2
SLICE_TYPE_NAMES = {SLICE_B: "B", SLICE_P: "P", SLICE_I: "I"}

PROFILE_NAMES = {
    1: "Main",
    2: "Main 10",
    3: "Main Still Picture",
    4: "Format range extensions",
    5: "High throughput",
    9: "Screen content coding",
}
CHROMA_FORMAT_NAMES = {0: "4:0:0", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}


class RtpError(ValueError):
    """A packet that is not a well-formed RTP version 2 packet."""


class BitstreamError(ValueError):
    """A NAL unit that could not be parsed (truncated, invalid, or missing its parameter sets)."""


# ---------------------------------------------------------------------------------------------------------------------
# RTP


class RtpHeader(NamedTuple):
    """The fixed RTP header (RFC 3550) plus where the payload is: packet[payload_offset:payload_end]."""

    marker: bool
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    csrcs: tuple[int, ...]
    extension_profile: int | None  # None without a header extension
    extension_length: int  # bytes of extension data after its 4-byte header
    padding: int  # padding bytes at the end of the packet
    payload_offset: int
    payload_end: int


_RTP_FIXED = struct.Struct("!BBHII")


def parse_rtp_header(packet: bytes, *, prefix: bool = False) -> RtpHeader:
    """Parse the RTP header of `packet`, including CSRCs, a header extension and padding. Raises RtpError.

    With prefix=True, `packet` is only the first bytes of a packet (as rtp_capture.py keeps them): padding, which is
    counted by the packet's last byte, is then not looked at.
    """
    if len(packet) < RTP_HEADER_SIZE:
        raise RtpError(f"{len(packet)} bytes is shorter than an RTP header")
    b0, b1, sequence, timestamp, ssrc = _RTP_FIXED.unpack_from(packet, 0)
    if b0 >> 6 != RTP_VERSION:
        raise RtpError(f"RTP version {b0 >> 6}")
    csrc_count = b0 & 0x0F
    offset = RTP_HEADER_SIZE + 4 * csrc_count
    if len(packet) < offset:
        raise RtpError(f"{csrc_count} CSRCs do not fit in {len(packet)} bytes")
    csrcs = struct.unpack_from(f"!{csrc_count}I", packet, RTP_HEADER_SIZE) if csrc_count else ()
    extension_profile = None
    extension_length = 0
    if b0 & 0x10:
        if len(packet) < offset + 4:
            raise RtpError("header extension does not fit")
        extension_profile, words = struct.unpack_from("!HH", packet, offset)
        extension_length = 4 * words
        offset += 4 + extension_length
        if len(packet) < offset:
            raise RtpError("header extension does not fit")
    end = len(packet)
    padding = 0
    if b0 & 0x20 and not prefix:
        padding = packet[-1]
        if padding == 0 or end - padding < offset:
            raise RtpError(f"invalid padding length {padding}")
        end -= padding
    return RtpHeader(
        marker=bool(b1 & 0x80),
        payload_type=b1 & 0x7F,
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc,
        csrcs=tuple(csrcs),
        extension_profile=extension_profile,
        extension_length=extension_length,
        padding=padding,
        payload_offset=offset,
        payload_end=end,
    )


def payload_type_of(payload: bytes) -> int:
    """The NAL unit type field of an RTP payload header: a NAL unit type (0-47), AP, FU or PACI."""
    return (payload[0] >> 1) & 0x3F


class Depacketizer:
    """Turns the payloads of one RTP stream, in arrival order, back into NAL units (RFC 7798).

    feed() returns the NAL units a packet completes: one for a single NAL unit packet, all of them for an aggregation
    packet, the reassembled unit for the last fragment of a fragmented one. A fragmented unit with a missing piece (a
    sequence gap inside it, a middle or end fragment without its start, a new start before the end) is dropped and
    counted once in `fu_dropped`. `packets` counts packets by kind: "single", "ap", "fu", "paci", "unknown" (types
    51-63), "invalid" (malformed).
    """

    def __init__(self, *, donl: bool = False):
        self.donl = donl
        self.packets: collections.Counter[str] = collections.Counter()
        self.fu_dropped = 0
        self._fu_parts: list[bytes] | None = None
        self._fu_skipping = False  # inside a unit that is already counted as dropped
        self._last_sequence: int | None = None

    def feed(self, header: RtpHeader, payload: bytes) -> list[bytes]:
        follows = self._last_sequence is not None and header.sequence == (self._last_sequence + 1) & 0xFFFF
        self._last_sequence = header.sequence
        if len(payload) < 2 or payload[0] & 0x80:  # no payload header, or the forbidden (F) bit set
            self.packets["invalid"] += 1
            self._end_fragmented_unit()
            return []
        kind = payload_type_of(payload)
        if kind == FU:
            return self._feed_fu(payload, follows=follows)
        self._end_fragmented_unit()
        if kind == AP:
            self.packets["ap"] += 1
            return self._feed_ap(payload)
        if kind == PACI:
            self.packets["paci"] += 1
            return []
        if kind > PACI:
            self.packets["unknown"] += 1
            return []
        self.packets["single"] += 1
        if self.donl:  # DONL sits between the NAL unit header and the payload
            return [payload[:2] + payload[4:]]
        return [payload]

    def flush(self) -> None:
        """End of stream: an unfinished fragmented unit counts as dropped."""
        self._end_fragmented_unit()

    def _end_fragmented_unit(self) -> None:
        if self._fu_parts is not None:
            self.fu_dropped += 1
        self._fu_parts = None
        self._fu_skipping = False

    def _feed_fu(self, payload: bytes, *, follows: bool) -> list[bytes]:
        if len(payload) < 3:
            self.packets["invalid"] += 1
            self._end_fragmented_unit()
            return []
        fu_header = payload[2]
        start, end, nal_type = fu_header & 0x80, fu_header & 0x40, fu_header & 0x3F
        if start and end:  # not allowed by RFC 7798
            self.packets["invalid"] += 1
            self._end_fragmented_unit()
            return []
        self.packets["fu"] += 1
        if start:
            self._end_fragmented_unit()
            # The NAL unit header: F, LayerId and TID from the payload header, the type from the FU header.
            nal_header = bytes(((payload[0] & 0x81) | (nal_type << 1), payload[1]))
            self._fu_parts = [nal_header, payload[5:] if self.donl else payload[3:]]
            return []
        if self._fu_parts is None or not follows:
            # A piece without the start of its unit, or after a lost packet: the unit cannot be rebuilt.
            if self._fu_parts is not None or not self._fu_skipping:
                self.fu_dropped += 1
            self._fu_parts = None
            self._fu_skipping = not end
            return []
        self._fu_parts.append(payload[3:])
        if not end:
            return []
        nal = b"".join(self._fu_parts)
        self._fu_parts = None
        return [nal]

    def _feed_ap(self, payload: bytes) -> list[bytes]:
        nals = []
        pos = 4 if self.donl else 2  # payload header, then DONL of the first unit
        while pos < len(payload):
            if nals and self.donl:
                pos += 1  # DOND
            if pos + 2 > len(payload):
                self.packets["invalid"] += 1
                break
            (size,) = struct.unpack_from("!H", payload, pos)
            pos += 2
            if size < 2 or pos + size > len(payload):
                self.packets["invalid"] += 1
                break
            nals.append(payload[pos : pos + size])
            pos += size
        return nals


class RtpAccessUnit(NamedTuple):
    """The NAL units that share one RTP timestamp, with the arrival (or send) times of their first and last packet."""

    timestamp: int
    nals: list[bytes]
    packets: int
    marker: bool  # ended by a marker bit; False when it ended because the timestamp changed (marker packet lost)
    first_time_ns: int
    last_time_ns: int


class RtpAccessUnits:
    """Groups depacketized NAL units into access units: one RTP timestamp each, ended by the marker bit."""

    def __init__(self):
        self._timestamp: int | None = None
        self._nals: list[bytes] = []
        self._packets = 0
        self._first_ns = 0
        self._last_ns = 0

    def add(self, header: RtpHeader, nals: list[bytes], time_ns: int) -> list[RtpAccessUnit]:
        """Add one packet's NAL units; returns the access units it completes (0, 1, or 2 when a marker was lost)."""
        done = []
        if self._timestamp is not None and header.timestamp != self._timestamp:
            done.append(self._finish(marker=False))
        if self._timestamp is None:
            self._timestamp = header.timestamp
            self._first_ns = time_ns
        self._nals.extend(nals)
        self._packets += 1
        self._last_ns = time_ns
        if header.marker:
            done.append(self._finish(marker=True))
        return done

    def flush(self) -> RtpAccessUnit | None:
        return None if self._timestamp is None else self._finish(marker=False)

    def _finish(self, *, marker: bool) -> RtpAccessUnit:
        assert self._timestamp is not None
        unit = RtpAccessUnit(self._timestamp, self._nals, self._packets, marker, self._first_ns, self._last_ns)
        self._timestamp = None
        self._nals = []
        self._packets = 0
        return unit


# ---------------------------------------------------------------------------------------------------------------------
# NAL units and Annex-B


def nal_type(nal: bytes) -> int:
    return (nal[0] >> 1) & 0x3F


def nal_layer_id(nal: bytes) -> int:
    return ((nal[0] & 1) << 5) | (nal[1] >> 3)


def nal_temporal_id(nal: bytes) -> int:
    return (nal[1] & 7) - 1


def nal_type_name(nal_unit_type: int) -> str:
    return NAL_TYPE_NAMES.get(nal_unit_type, f"type{nal_unit_type}")


def is_vcl(nal_unit_type: int) -> bool:
    return nal_unit_type < 32


def is_irap(nal_unit_type: int) -> bool:
    """IRAP pictures (BLA, IDR, CRA and reserved IRAP types): decoding can start here."""
    return BLA_W_LP <= nal_unit_type <= RSV_IRAP_VCL23


def is_idr(nal_unit_type: int) -> bool:
    return nal_unit_type in (IDR_W_RADL, IDR_N_LP)


def first_slice_segment_in_pic(nal: bytes) -> bool:
    """For a VCL NAL unit: whether it starts a new picture (the first bit after the 2-byte NAL unit header)."""
    return len(nal) > 2 and bool(nal[2] & 0x80)


START_CODE = b"\x00\x00\x01"


def split_annexb(data: bytes) -> list[bytes]:
    """The NAL units of an Annex-B byte stream (start codes and trailing zero bytes removed)."""
    nals = []
    start = data.find(START_CODE)
    while start >= 0:
        start += 3
        end = data.find(START_CODE, start)
        nal = (data[start:] if end < 0 else data[start:end]).rstrip(b"\x00")
        if nal:
            nals.append(nal)
        start = end
    return nals


def iter_annexb(stream: BinaryIO, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Like split_annexb, reading `stream` in chunks so large files need not fit in memory."""
    buf = b""
    nal_start: int | None = None
    search_from = 0
    while True:
        chunk = stream.read(chunk_size)
        buf += chunk
        while (found := buf.find(START_CODE, search_from)) >= 0:
            if nal_start is not None:
                nal = buf[nal_start:found].rstrip(b"\x00")  # a NAL unit never ends in 0x00; these are zero bytes
                if nal:
                    yield nal
            nal_start = search_from = found + 3
        if not chunk:
            break
        # Keep only the unfinished NAL unit. A start code may straddle the chunk boundary, so look again from 2 bytes
        # before the end.
        if nal_start is None:
            buf = buf[-2:]
        else:
            buf = buf[nal_start:]
            nal_start = 0
        search_from = max(0, len(buf) - 2)
    if nal_start is not None:
        nal = buf[nal_start:].rstrip(b"\x00")
        if nal:
            yield nal


def to_annexb(nals: Iterable[bytes]) -> bytes:
    """An Annex-B byte stream with a 4-byte start code before every NAL unit."""
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


class AnnexBAccessUnits:
    """Groups the NAL units of a byte stream into access units by the rules of H.265 section 7.4.2.4.4: after a
    picture's slices, an AUD, VPS, SPS, PPS, prefix SEI or a slice with first_slice_segment_in_pic_flag starts the
    next access unit."""

    _STARTS_UNIT = frozenset((AUD_NUT, VPS_NUT, SPS_NUT, PPS_NUT, PREFIX_SEI_NUT, 41, 42, 43, 44, *range(48, 56)))

    def __init__(self):
        self._nals: list[bytes] = []
        self._has_vcl = False

    def add(self, nal: bytes) -> list[bytes] | None:
        """Add one NAL unit; returns the previous access unit when this one starts a new one."""
        kind = nal_type(nal)
        vcl = is_vcl(kind)
        done = None
        if self._has_vcl and (kind in self._STARTS_UNIT or (vcl and first_slice_segment_in_pic(nal))):
            done = self._nals
            self._nals = []
            self._has_vcl = False
        self._nals.append(nal)
        self._has_vcl = self._has_vcl or vcl
        return done

    def flush(self) -> list[bytes] | None:
        """End of stream: the last access unit, if any."""
        done = self._nals or None
        self._nals = []
        self._has_vcl = False
        return done


# ---------------------------------------------------------------------------------------------------------------------
# Bit-level parsing


def unescape_rbsp(data: bytes) -> bytes:
    """Remove emulation prevention bytes: in a NAL unit every 0x000003 is 0x0000 followed by one."""
    return data.replace(b"\x00\x00\x03", b"\x00\x00")


class BitReader:
    """Reads an RBSP most significant bit first: fixed-length u(n) and exp-Golomb ue(v)/se(v) codes."""

    def __init__(self, data: bytes):
        self._value = int.from_bytes(data, "big")
        self._size = 8 * len(data)
        self.pos = 0

    @property
    def bits_left(self) -> int:
        return self._size - self.pos

    def u(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > self._size:
            raise BitstreamError(f"read of {n} bits at bit {self.pos} passes the end ({self._size} bits)")
        self.pos += n
        return (self._value >> (self._size - self.pos)) & ((1 << n) - 1)

    def flag(self) -> bool:
        return self.u(1) == 1

    def skip(self, n: int) -> None:
        if self.pos + n > self._size:
            raise BitstreamError(f"skip of {n} bits at bit {self.pos} passes the end ({self._size} bits)")
        self.pos += n

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise BitstreamError("exp-Golomb code longer than 32 bits")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self) -> int:
        code = self.ue()
        return (code + 1) // 2 if code & 1 else -(code // 2)


# ---------------------------------------------------------------------------------------------------------------------
# Parameter sets (H.265 section 7.3.2)


@dataclasses.dataclass
class ProfileTierLevel:
    profile_space: int
    tier: int  # 0 Main tier, 1 High tier
    profile_idc: int
    compatibility_flags: int  # general_profile_compatibility_flag[0..31] as a 32-bit number, flag 0 first
    progressive_source: bool
    interlaced_source: bool
    level_idc: int  # 30 x level number, e.g. 150 for level 5

    @property
    def profile_name(self) -> str:
        return PROFILE_NAMES.get(self.profile_idc, f"profile {self.profile_idc}")

    @property
    def level(self) -> float:
        return self.level_idc / 30


def _parse_profile_tier_level(r: BitReader, max_sub_layers_minus1: int) -> ProfileTierLevel:
    profile_space = r.u(2)
    tier = r.u(1)
    profile_idc = r.u(5)
    compatibility = r.u(32)
    progressive = r.flag()
    interlaced = r.flag()
    r.skip(2 + 43 + 1)  # non_packed, frame_only, 43 constraint bits, inbld/reserved
    level_idc = r.u(8)
    sub_layer_profile = []
    sub_layer_level = []
    for _ in range(max_sub_layers_minus1):
        sub_layer_profile.append(r.flag())
        sub_layer_level.append(r.flag())
    if max_sub_layers_minus1 > 0:
        r.skip(2 * (8 - max_sub_layers_minus1))  # reserved_zero_2bits
    for profile_present, level_present in zip(sub_layer_profile, sub_layer_level, strict=True):
        if profile_present:
            r.skip(88)
        if level_present:
            r.skip(8)
    return ProfileTierLevel(profile_space, tier, profile_idc, compatibility, progressive, interlaced, level_idc)


def _parse_sub_layer_ordering(r: BitReader, max_sub_layers_minus1: int) -> tuple[list[int], list[int], list[int]]:
    """(max_dec_pic_buffering_minus1, max_num_reorder_pics, max_latency_increase_plus1), one entry per sub-layer."""
    present = r.flag()
    first = 0 if present else max_sub_layers_minus1
    buffering, reorder, latency = [], [], []
    for _ in range(first, max_sub_layers_minus1 + 1):
        buffering.append(r.ue())
        reorder.append(r.ue())
        latency.append(r.ue())
    return buffering, reorder, latency


def _skip_scaling_list_data(r: BitReader) -> None:
    for size_id in range(4):
        for _matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            if not r.flag():  # scaling_list_pred_mode_flag
                r.ue()  # scaling_list_pred_matrix_id_delta
                continue
            coefficients = min(64, 1 << (4 + (size_id << 1)))
            if size_id > 1:
                r.se()  # scaling_list_dc_coef_minus8
            for _ in range(coefficients):
                r.se()  # scaling_list_delta_coef


@dataclasses.dataclass
class Vps:
    vps_id: int
    max_layers: int
    max_sub_layers: int
    temporal_id_nesting: bool
    profile_tier_level: ProfileTierLevel
    max_dec_pic_buffering_minus1: list[int]
    max_num_reorder_pics: list[int]


def parse_vps(nal: bytes) -> Vps:
    r = BitReader(unescape_rbsp(nal[2:]))
    try:
        vps_id = r.u(4)
        r.skip(2)  # vps_base_layer_internal_flag, vps_base_layer_available_flag
        max_layers = r.u(6) + 1
        max_sub_layers_minus1 = r.u(3)
        nesting = r.flag()
        r.skip(16)  # vps_reserved_0xffff_16bits
        ptl = _parse_profile_tier_level(r, max_sub_layers_minus1)
        buffering, reorder, _latency = _parse_sub_layer_ordering(r, max_sub_layers_minus1)
    except BitstreamError as exc:
        raise BitstreamError(f"VPS: {exc}") from exc
    return Vps(vps_id, max_layers, max_sub_layers_minus1 + 1, nesting, ptl, buffering, reorder)


@dataclasses.dataclass
class Vui:
    """The start of vui_parameters(): colour description and timing (the HRD and what follows are not parsed)."""

    video_full_range: bool | None = None  # None when video_signal_type_present_flag is 0
    colour_primaries: int | None = None
    transfer_characteristics: int | None = None
    matrix_coefficients: int | None = None
    num_units_in_tick: int | None = None
    time_scale: int | None = None

    @property
    def fps(self) -> float | None:
        if not self.num_units_in_tick or not self.time_scale:
            return None
        return self.time_scale / self.num_units_in_tick


def _parse_vui(r: BitReader) -> Vui:
    vui = Vui()
    if r.flag() and r.u(8) == 255:  # aspect_ratio_info_present_flag, aspect_ratio_idc == EXTENDED_SAR
        r.skip(32)  # sar_width, sar_height
    if r.flag():  # overscan_info_present_flag
        r.skip(1)
    if r.flag():  # video_signal_type_present_flag
        r.skip(3)  # video_format
        vui.video_full_range = r.flag()
        if r.flag():  # colour_description_present_flag
            vui.colour_primaries = r.u(8)
            vui.transfer_characteristics = r.u(8)
            vui.matrix_coefficients = r.u(8)
    if r.flag():  # chroma_loc_info_present_flag
        r.ue()
        r.ue()
    r.skip(3)  # neutral_chroma_indication_flag, field_seq_flag, frame_field_info_present_flag
    if r.flag():  # default_display_window_flag
        for _ in range(4):
            r.ue()
    if r.flag():  # vui_timing_info_present_flag
        vui.num_units_in_tick = r.u(32)
        vui.time_scale = r.u(32)
    return vui


@dataclasses.dataclass
class Sps:
    sps_id: int
    vps_id: int
    max_sub_layers: int
    profile_tier_level: ProfileTierLevel
    chroma_format_idc: int
    separate_colour_plane: bool
    width: int  # pic_width_in_luma_samples (coded size, a multiple of the minimum coding block)
    height: int
    conformance_window: tuple[int, int, int, int]  # left, right, top, bottom offsets in chroma sample units
    bit_depth_luma: int
    bit_depth_chroma: int
    log2_max_pic_order_cnt_lsb: int
    max_dec_pic_buffering_minus1: list[int]  # per sub-layer (only the highest when not signalled per sub-layer)
    max_num_reorder_pics: list[int]
    max_latency_increase_plus1: list[int]
    log2_min_coding_block_size: int
    log2_ctb_size: int
    log2_min_transform_block_size: int
    log2_max_transform_block_size: int
    scaling_list_enabled: bool
    amp_enabled: bool
    sample_adaptive_offset_enabled: bool
    pcm_enabled: bool
    num_short_term_ref_pic_sets: int
    long_term_ref_pics_present: bool
    temporal_mvp_enabled: bool
    strong_intra_smoothing_enabled: bool
    vui: Vui | None

    @property
    def ctb_size(self) -> int:
        return 1 << self.log2_ctb_size

    @property
    def width_in_ctbs(self) -> int:
        return -(-self.width // self.ctb_size)

    @property
    def height_in_ctbs(self) -> int:
        return -(-self.height // self.ctb_size)

    @property
    def slice_segment_address_bits(self) -> int:
        """Bits of slice_segment_address: Ceil(Log2(PicSizeInCtbsY))."""
        return (self.width_in_ctbs * self.height_in_ctbs - 1).bit_length()

    @property
    def cropped_size(self) -> tuple[int, int]:
        """(width, height) after the conformance window, i.e. the size a decoder outputs."""
        sub_width = 2 if self.chroma_format_idc in (1, 2) and not self.separate_colour_plane else 1
        sub_height = 2 if self.chroma_format_idc == 1 and not self.separate_colour_plane else 1
        left, right, top, bottom = self.conformance_window
        return self.width - sub_width * (left + right), self.height - sub_height * (top + bottom)


def _parse_st_ref_pic_set(r: BitReader, index: int, num_delta_pocs: list[int]) -> None:
    """Skip st_ref_pic_set(index) of an SPS, recording NumDeltaPocs[index] (needed by later sets)."""
    if index != 0 and r.flag():  # inter_ref_pic_set_prediction_flag (delta_idx_minus1 only exists in slice headers)
        r.skip(1)  # delta_rps_sign
        r.ue()  # abs_delta_rps_minus1
        count = 0
        for _ in range(num_delta_pocs[index - 1] + 1):
            used_by_curr_pic = r.flag()
            use_delta = used_by_curr_pic or r.flag()  # use_delta_flag is inferred 1 when not present
            count += use_delta
        num_delta_pocs.append(count)
        return
    negative = r.ue()
    positive = r.ue()
    for _ in range(negative + positive):
        r.ue()  # delta_poc_s0/s1_minus1
        r.skip(1)  # used_by_curr_pic_s0/s1_flag
    num_delta_pocs.append(negative + positive)


def parse_sps(nal: bytes) -> Sps:
    r = BitReader(unescape_rbsp(nal[2:]))
    try:
        return _parse_sps(r)
    except BitstreamError as exc:
        raise BitstreamError(f"SPS: {exc}") from exc


def _parse_sps(r: BitReader) -> Sps:
    vps_id = r.u(4)
    max_sub_layers_minus1 = r.u(3)
    r.skip(1)  # sps_temporal_id_nesting_flag
    ptl = _parse_profile_tier_level(r, max_sub_layers_minus1)
    sps_id = r.ue()
    chroma_format_idc = r.ue()
    separate_colour_plane = r.flag() if chroma_format_idc == 3 else False
    width = r.ue()
    height = r.ue()
    window = (r.ue(), r.ue(), r.ue(), r.ue()) if r.flag() else (0, 0, 0, 0)
    bit_depth_luma = r.ue() + 8
    bit_depth_chroma = r.ue() + 8
    log2_max_poc_lsb = r.ue() + 4
    buffering, reorder, latency = _parse_sub_layer_ordering(r, max_sub_layers_minus1)
    log2_min_cb = r.ue() + 3
    log2_ctb = log2_min_cb + r.ue()
    log2_min_tb = r.ue() + 2
    log2_max_tb = log2_min_tb + r.ue()
    r.ue()  # max_transform_hierarchy_depth_inter
    r.ue()  # max_transform_hierarchy_depth_intra
    scaling_list_enabled = r.flag()
    if scaling_list_enabled and r.flag():  # sps_scaling_list_data_present_flag
        _skip_scaling_list_data(r)
    amp = r.flag()
    sao = r.flag()
    pcm = r.flag()
    if pcm:
        r.skip(8)  # pcm_sample_bit_depth_luma_minus1, pcm_sample_bit_depth_chroma_minus1
        r.ue()
        r.ue()
        r.skip(1)  # pcm_loop_filter_disabled_flag
    num_st_rps = r.ue()
    num_delta_pocs: list[int] = []
    for index in range(num_st_rps):
        _parse_st_ref_pic_set(r, index, num_delta_pocs)
    long_term = r.flag()
    if long_term:
        for _ in range(r.ue()):  # num_long_term_ref_pics_sps
            r.skip(log2_max_poc_lsb + 1)  # lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps_flag
    temporal_mvp = r.flag()
    strong_intra_smoothing = r.flag()
    vui = _parse_vui(r) if r.flag() else None
    return Sps(
        sps_id=sps_id,
        vps_id=vps_id,
        max_sub_layers=max_sub_layers_minus1 + 1,
        profile_tier_level=ptl,
        chroma_format_idc=chroma_format_idc,
        separate_colour_plane=separate_colour_plane,
        width=width,
        height=height,
        conformance_window=window,
        bit_depth_luma=bit_depth_luma,
        bit_depth_chroma=bit_depth_chroma,
        log2_max_pic_order_cnt_lsb=log2_max_poc_lsb,
        max_dec_pic_buffering_minus1=buffering,
        max_num_reorder_pics=reorder,
        max_latency_increase_plus1=latency,
        log2_min_coding_block_size=log2_min_cb,
        log2_ctb_size=log2_ctb,
        log2_min_transform_block_size=log2_min_tb,
        log2_max_transform_block_size=log2_max_tb,
        scaling_list_enabled=scaling_list_enabled,
        amp_enabled=amp,
        sample_adaptive_offset_enabled=sao,
        pcm_enabled=pcm,
        num_short_term_ref_pic_sets=num_st_rps,
        long_term_ref_pics_present=long_term,
        temporal_mvp_enabled=temporal_mvp,
        strong_intra_smoothing_enabled=strong_intra_smoothing,
        vui=vui,
    )


@dataclasses.dataclass
class Pps:
    pps_id: int
    sps_id: int
    dependent_slice_segments_enabled: bool
    output_flag_present: bool
    num_extra_slice_header_bits: int
    sign_data_hiding_enabled: bool
    cabac_init_present: bool
    init_qp: int
    constrained_intra_pred: bool
    transform_skip_enabled: bool
    cu_qp_delta_enabled: bool
    weighted_pred: bool
    weighted_bipred: bool
    transquant_bypass_enabled: bool
    tiles_enabled: bool
    entropy_coding_sync_enabled: bool  # wavefront parallel processing (WPP)
    num_tile_columns: int
    num_tile_rows: int
    uniform_tile_spacing: bool
    loop_filter_across_slices_enabled: bool
    deblocking_filter_override_enabled: bool
    deblocking_filter_disabled: bool  # pps_deblocking_filter_disabled_flag (slices may override when enabled)


def parse_pps(nal: bytes) -> Pps:
    r = BitReader(unescape_rbsp(nal[2:]))
    try:
        return _parse_pps(r)
    except BitstreamError as exc:
        raise BitstreamError(f"PPS: {exc}") from exc


def _parse_pps(r: BitReader) -> Pps:
    pps_id = r.ue()
    sps_id = r.ue()
    dependent_slices = r.flag()
    output_flag_present = r.flag()
    extra_bits = r.u(3)
    sign_data_hiding = r.flag()
    cabac_init_present = r.flag()
    r.ue()  # num_ref_idx_l0_default_active_minus1
    r.ue()  # num_ref_idx_l1_default_active_minus1
    init_qp = 26 + r.se()
    constrained_intra_pred = r.flag()
    transform_skip = r.flag()
    cu_qp_delta = r.flag()
    if cu_qp_delta:
        r.ue()  # diff_cu_qp_delta_depth
    r.se()  # pps_cb_qp_offset
    r.se()  # pps_cr_qp_offset
    r.skip(1)  # pps_slice_chroma_qp_offsets_present_flag
    weighted_pred = r.flag()
    weighted_bipred = r.flag()
    transquant_bypass = r.flag()
    tiles = r.flag()
    wpp = r.flag()
    columns = rows = 1
    uniform = True
    if tiles:
        columns = r.ue() + 1
        rows = r.ue() + 1
        uniform = r.flag()
        if not uniform:
            for _ in range(columns - 1 + rows - 1):
                r.ue()  # column_width_minus1, row_height_minus1
        r.skip(1)  # loop_filter_across_tiles_enabled_flag
    across_slices = r.flag()
    override_enabled = deblocking_disabled = False
    if r.flag():  # deblocking_filter_control_present_flag
        override_enabled = r.flag()
        deblocking_disabled = r.flag()
        if not deblocking_disabled:
            r.se()  # pps_beta_offset_div2
            r.se()  # pps_tc_offset_div2
    return Pps(
        pps_id=pps_id,
        sps_id=sps_id,
        dependent_slice_segments_enabled=dependent_slices,
        output_flag_present=output_flag_present,
        num_extra_slice_header_bits=extra_bits,
        sign_data_hiding_enabled=sign_data_hiding,
        cabac_init_present=cabac_init_present,
        init_qp=init_qp,
        constrained_intra_pred=constrained_intra_pred,
        transform_skip_enabled=transform_skip,
        cu_qp_delta_enabled=cu_qp_delta,
        weighted_pred=weighted_pred,
        weighted_bipred=weighted_bipred,
        transquant_bypass_enabled=transquant_bypass,
        tiles_enabled=tiles,
        entropy_coding_sync_enabled=wpp,
        num_tile_columns=columns,
        num_tile_rows=rows,
        uniform_tile_spacing=uniform,
        loop_filter_across_slices_enabled=across_slices,
        deblocking_filter_override_enabled=override_enabled,
        deblocking_filter_disabled=deblocking_disabled,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Slice segment headers (H.265 section 7.3.6.1)


class SliceHeader(NamedTuple):
    first_slice_segment_in_pic: bool
    pps_id: int
    dependent: bool  # a dependent slice segment: slice_type and the rest are those of the preceding segment
    segment_address: int  # first CTB of the segment, in raster scan
    slice_type: int | None  # SLICE_B, SLICE_P, SLICE_I; None for a dependent segment
    pic_order_cnt_lsb: int | None  # None for IDR pictures (their POC is 0) and dependent segments


# A slice header up to slice_pic_order_cnt_lsb is under ~12 bytes; parse only a prefix of the NAL unit.
_SLICE_HEADER_BYTES = 48


def parse_slice_header(nal: bytes, pps_by_id: dict[int, Pps], sps_by_id: dict[int, Sps]) -> SliceHeader:
    """Parse a VCL NAL unit's slice segment header up to slice_pic_order_cnt_lsb with the active parameter sets."""
    kind = nal_type(nal)
    r = BitReader(unescape_rbsp(nal[2 : 2 + _SLICE_HEADER_BYTES]))
    try:
        first = r.flag()
        if BLA_W_LP <= kind <= RSV_IRAP_VCL23:
            r.skip(1)  # no_output_of_prior_pics_flag
        pps_id = r.ue()
        pps = pps_by_id.get(pps_id)
        if pps is None:
            raise BitstreamError(f"slice refers to PPS {pps_id}, not seen yet")
        sps = sps_by_id.get(pps.sps_id)
        if sps is None:
            raise BitstreamError(f"PPS {pps_id} refers to SPS {pps.sps_id}, not seen yet")
        dependent = False
        address = 0
        if not first:
            if pps.dependent_slice_segments_enabled:
                dependent = r.flag()
            address = r.u(sps.slice_segment_address_bits)
        if dependent:
            return SliceHeader(
                first, pps_id, dependent=True, segment_address=address, slice_type=None, pic_order_cnt_lsb=None
            )
        r.skip(pps.num_extra_slice_header_bits)  # slice_reserved_flag
        slice_type = r.ue()
        if slice_type not in SLICE_TYPE_NAMES:
            raise BitstreamError(f"slice_type {slice_type}")
        if pps.output_flag_present:
            r.skip(1)  # pic_output_flag
        if sps.separate_colour_plane:
            r.skip(2)  # colour_plane_id
        poc_lsb = None if is_idr(kind) else r.u(sps.log2_max_pic_order_cnt_lsb)
    except BitstreamError as exc:
        raise BitstreamError(f"slice header ({nal_type_name(kind)}): {exc}") from exc
    return SliceHeader(
        first, pps_id, dependent=False, segment_address=address, slice_type=slice_type, pic_order_cnt_lsb=poc_lsb
    )
