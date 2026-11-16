#!/usr/bin/env python3
"""Summarize an H.265 stream: a .rtpcap capture or fixture, or an Annex-B .h265 file.

    python3 stream_info.py robot.rtpcap [--json robot_info.json] [--write-h265 robot.h265]
    python3 stream_info.py clip.h265 --fps 45

Reports what decides how a CPU decoder copes with the stream: for a .rtpcap the packet statistics of rtp_capture.py
(packets, access units, frame rate from the RTP timestamps, bitrate, packet sizes, single NAL unit / AP / FU shares,
sequence gaps) and its packets_sha256; for both inputs the bitstream: profile, level, size, bit depth, chroma format,
CTB size, SAO, colour description, WPP (entropy_coding_sync), tiles, dependent slice segments, slice segments per
access unit, slice types (I/P/B) and whether pictures are reordered, IRAP/IDR intervals in frames and seconds, and
how the VPS/SPS/PPS are repeated. --write-h265 writes the depacketized stream as Annex-B (for offline decoding, e.g.
compare_decoders.py). With --drop-incomplete-aus (a .rtpcap only) the access units a lost or reordered packet may have
cut short are left out of that file, so that two decoders compared on it see the same, whole pictures: an access unit
whose RTP timestamp is on either side of a sequence gap or a late packet, or of a fragmented unit the depacketizer had
to drop, one ended without its marker bit, and one without any slice; their VPS/SPS/PPS are still written. The summary
counts what was written and left out (and lists the first left-out units). Pure Python, no gi.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable
import itertools
import json
import sys
from typing import BinaryIO

import h265_rtp
import rtp_capture
import rtpcap

PARAMETER_SET_NAMES = {h265_rtp.VPS_NUT: "VPS", h265_rtp.SPS_NUT: "SPS", h265_rtp.PPS_NUT: "PPS"}
DROPPED_LISTED_MAX = 100
# Pictures that do not update prevTid0Pic for picture order count derivation (H.265 8.3.1): RADL, RASL and
# sub-layer non-reference pictures.
_NOT_PREV_TID0 = frozenset((0, 2, 4, 6, 7, 8, 9, 10, 12, 14))


class BitstreamStats:
    """Follows the access units of one stream: parameter sets, slice headers, picture order, IRAP spacing."""

    def __init__(self):
        self.vps: dict[int, h265_rtp.Vps] = {}
        self.sps: dict[int, h265_rtp.Sps] = {}
        self.pps: dict[int, h265_rtp.Pps] = {}
        self.access_units = 0
        self.nals = 0
        self.es_bytes = 0
        self.nal_types: collections.Counter[str] = collections.Counter()
        self.slices_per_au: collections.Counter[int] = collections.Counter()
        self.dependent_segments = 0
        self.slice_types: collections.Counter[str] = collections.Counter()
        self.au_without_slices = 0
        self.irap_indices: list[int] = []
        self.idr_count = 0
        self.cra_count = 0
        self.irap_au_bytes: list[int] = []
        self.other_au_bytes: list[int] = []
        self.parameter_set_copies: collections.Counter[str] = collections.Counter()
        self.distinct_parameter_sets: dict[str, set[bytes]] = collections.defaultdict(set)
        self.parameter_sets_per_irap: collections.Counter[str] = collections.Counter()
        self.parameter_sets_outside_irap = 0
        self.reordered_pictures = 0
        self.errors: collections.Counter[str] = collections.Counter()
        self._prev_poc_msb = 0
        self._prev_poc_lsb = 0
        self._last_poc: int | None = None

    def add_access_unit(self, nals: list[bytes]) -> None:
        index = self.access_units
        self.access_units += 1
        size = 0
        slices = 0
        parameter_sets: collections.Counter[str] = collections.Counter()
        irap_type = None
        for nal in nals:
            size += len(nal)
            self.nals += 1
            if len(nal) < 2:
                self.errors["NAL unit shorter than its header"] += 1
                continue
            kind = h265_rtp.nal_type(nal)
            self.nal_types[h265_rtp.nal_type_name(kind)] += 1
            if kind in PARAMETER_SET_NAMES:
                name = PARAMETER_SET_NAMES[kind]
                parameter_sets[name] += 1
                self.parameter_set_copies[name] += 1
                self.distinct_parameter_sets[name].add(nal)
                self._parse_parameter_set(kind, nal)
            elif h265_rtp.is_vcl(kind):
                slices += 1
                if h265_rtp.is_irap(kind):
                    irap_type = kind
                self._parse_slice(kind, nal)
        self.es_bytes += size
        self.slices_per_au[slices] += 1
        if slices == 0:
            self.au_without_slices += 1
        if irap_type is not None:
            self.irap_indices.append(index)
            self.idr_count += h265_rtp.is_idr(irap_type)
            self.cra_count += irap_type == h265_rtp.CRA_NUT
            self.irap_au_bytes.append(size)
            key = "/".join(str(parameter_sets[name]) for name in ("VPS", "SPS", "PPS"))
            self.parameter_sets_per_irap[key] += 1
        else:
            self.other_au_bytes.append(size)
            self.parameter_sets_outside_irap += sum(parameter_sets.values())

    def _parse_parameter_set(self, kind: int, nal: bytes) -> None:
        try:
            if kind == h265_rtp.VPS_NUT:
                vps = h265_rtp.parse_vps(nal)
                self.vps[vps.vps_id] = vps
            elif kind == h265_rtp.SPS_NUT:
                sps = h265_rtp.parse_sps(nal)
                self.sps[sps.sps_id] = sps
            else:
                pps = h265_rtp.parse_pps(nal)
                self.pps[pps.pps_id] = pps
        except h265_rtp.BitstreamError as exc:
            self.errors[str(exc)] += 1

    def _parse_slice(self, kind: int, nal: bytes) -> None:
        try:
            header = h265_rtp.parse_slice_header(nal, self.pps, self.sps)
        except h265_rtp.BitstreamError as exc:
            self.errors[str(exc)] += 1
            return
        if header.dependent:
            self.dependent_segments += 1
            return
        self.slice_types[h265_rtp.SLICE_TYPE_NAMES[header.slice_type]] += 1
        if header.first_slice_segment_in_pic:
            self._track_picture_order(kind, nal, header)

    def _track_picture_order(self, kind: int, nal: bytes, header: h265_rtp.SliceHeader) -> None:
        """PicOrderCntVal as in H.265 8.3.1; a picture with a lower POC than the one decoded before it (in the same
        coded video sequence) is a reordered picture, i.e. the stream has B-frame style reordering."""
        if header.pic_order_cnt_lsb is None:  # IDR: POC 0, new coded video sequence
            poc_msb, poc_lsb, new_sequence = 0, 0, True
        else:
            pps = self.pps[header.pps_id]
            max_lsb = 1 << self.sps[pps.sps_id].log2_max_pic_order_cnt_lsb
            poc_lsb = header.pic_order_cnt_lsb
            # BLA, and a CRA that starts the stream, begin a new coded video sequence.
            new_sequence = h265_rtp.is_irap(kind) and (kind != h265_rtp.CRA_NUT or self._last_poc is None)
            if new_sequence:
                poc_msb = 0
            elif poc_lsb < self._prev_poc_lsb and self._prev_poc_lsb - poc_lsb >= max_lsb // 2:
                poc_msb = self._prev_poc_msb + max_lsb
            elif poc_lsb > self._prev_poc_lsb and poc_lsb - self._prev_poc_lsb > max_lsb // 2:
                poc_msb = self._prev_poc_msb - max_lsb
            else:
                poc_msb = self._prev_poc_msb
        poc = poc_msb + poc_lsb
        if not new_sequence and self._last_poc is not None and poc < self._last_poc:
            self.reordered_pictures += 1
        self._last_poc = poc
        if h265_rtp.nal_temporal_id(nal) == 0 and kind not in _NOT_PREV_TID0:
            self._prev_poc_msb, self._prev_poc_lsb = poc_msb, poc_lsb

    def summary(self, fps: float | None) -> dict:
        intervals = collections.Counter(b - a for a, b in itertools.pairwise(self.irap_indices))
        result = {
            "access_units": self.access_units,
            "nal_units": self.nals,
            "es_bytes": self.es_bytes,
            "fps": fps,
            "duration_s": round(self.access_units / fps, 3) if fps else None,
            "es_bitrate_mbps": round(8 * self.es_bytes * fps / self.access_units / 1e6, 3)
            if fps and self.access_units
            else None,
        }
        result.update(self._parameter_set_summary())
        result.update(
            {
                "slice_segments_per_au": {str(k): v for k, v in sorted(self.slices_per_au.items())},
                "dependent_slice_segments": self.dependent_segments,
                "slice_types": {name: self.slice_types[name] for name in ("I", "P", "B")},
                "b_slices": self.slice_types["B"],
                "reordered_pictures": self.reordered_pictures,
                "irap_count": len(self.irap_indices),
                "idr_count": self.idr_count,
                "cra_count": self.cra_count,
                "first_irap_au": self.irap_indices[0] if self.irap_indices else None,
                "irap_interval_frames": {str(k): v for k, v in sorted(intervals.items())},
                "irap_interval_s": {f"{k / fps:.4f}": v for k, v in sorted(intervals.items())} if fps else None,
                "parameter_sets": {
                    "copies": dict(sorted(self.parameter_set_copies.items())),
                    "distinct": {name: len(sets) for name, sets in sorted(self.distinct_parameter_sets.items())},
                    "per_irap_au_vps_sps_pps": dict(sorted(self.parameter_sets_per_irap.items())),
                    "in_other_aus": self.parameter_sets_outside_irap,
                },
                "irap_au_bytes": rtp_capture.percentiles(self.irap_au_bytes),
                "other_au_bytes": rtp_capture.percentiles(self.other_au_bytes),
                "nal_types": dict(sorted(self.nal_types.items())),
                "access_units_without_slices": self.au_without_slices,
                "errors": dict(self.errors.most_common(20)),
            }
        )
        return result

    def _parameter_set_summary(self) -> dict:
        if not self.sps or not self.pps:
            return {"sps": None, "pps": None}
        # The parameter sets in use at the end of the stream (fixtures and cameras keep one of each).
        pps = self.pps[max(self.pps)]
        sps = self.sps.get(pps.sps_id) or self.sps[max(self.sps)]
        ptl = sps.profile_tier_level
        width, height = sps.cropped_size
        vui = sps.vui
        return {
            "profile": ptl.profile_name,
            "tier": "High" if ptl.tier else "Main",
            "level": round(ptl.level, 2),
            "width": width,
            "height": height,
            "coded_width": sps.width,
            "coded_height": sps.height,
            "chroma_format": h265_rtp.CHROMA_FORMAT_NAMES.get(sps.chroma_format_idc, str(sps.chroma_format_idc)),
            "bit_depth_luma": sps.bit_depth_luma,
            "bit_depth_chroma": sps.bit_depth_chroma,
            "ctb_size": sps.ctb_size,
            "min_cb_size": 1 << sps.log2_min_coding_block_size,
            "ctbs_per_row": sps.width_in_ctbs,
            "ctb_rows": sps.height_in_ctbs,
            "sao": sps.sample_adaptive_offset_enabled,
            "amp": sps.amp_enabled,
            "pcm": sps.pcm_enabled,
            "scaling_lists": sps.scaling_list_enabled,
            "temporal_mvp": sps.temporal_mvp_enabled,
            "strong_intra_smoothing": sps.strong_intra_smoothing_enabled,
            "sps_max_dec_pic_buffering": [n + 1 for n in sps.max_dec_pic_buffering_minus1],
            "sps_max_num_reorder_pics": sps.max_num_reorder_pics,
            "vui": None if vui is None else {k: v for k, v in vars(vui).items() if v is not None} | {"fps": vui.fps},
            "wpp": pps.entropy_coding_sync_enabled,
            "tiles": [pps.num_tile_columns, pps.num_tile_rows] if pps.tiles_enabled else None,
            "uniform_tile_spacing": pps.uniform_tile_spacing if pps.tiles_enabled else None,
            "dependent_slice_segments_enabled": pps.dependent_slice_segments_enabled,
            "num_extra_slice_header_bits": pps.num_extra_slice_header_bits,
            "deblocking_disabled_in_pps": pps.deblocking_filter_disabled,
            "deblocking_override_enabled": pps.deblocking_filter_override_enabled,
            "loop_filter_across_slices": pps.loop_filter_across_slices_enabled,
            "sign_data_hiding": pps.sign_data_hiding_enabled,
            "transform_skip": pps.transform_skip_enabled,
            "cu_qp_delta": pps.cu_qp_delta_enabled,
            "weighted_pred": pps.weighted_pred,
            "init_qp": pps.init_qp,
            "vps_count": len(self.vps),
            "sps_count": len(self.sps),
            "pps_count": len(self.pps),
        }


class _Damage:
    """--drop-incomplete-aus: the RTP timestamps of the access units a lost or reordered packet may have cut short. A
    packet after a sequence gap, a late packet, and a packet at which the depacketizer dropped a fragmented unit mark
    their own timestamp and the previous packet's (the unit cut short may be the one before)."""

    def __init__(self):
        self.timestamps: set[int] = set()
        self._newest: int | None = None
        self._previous_timestamp: int | None = None
        self._fu_dropped = 0

    def packet(self, header: h265_rtp.RtpHeader, fu_dropped: int) -> None:
        damaged = fu_dropped != self._fu_dropped
        if self._newest is None:
            self._newest = header.sequence
        else:
            delta = (header.sequence - self._newest) & 0xFFFF
            if delta == 0 or delta >= 0x8000:  # late or duplicate
                damaged = True
            else:
                damaged = damaged or delta > 1
                self._newest = header.sequence
        if damaged:
            self.timestamps.add(header.timestamp)
            if self._previous_timestamp is not None:
                self.timestamps.add(self._previous_timestamp)
        self._fu_dropped = fu_dropped
        self._previous_timestamp = header.timestamp


class _AnnexBWriter:
    """Writes access units as Annex-B; with `damage`, one access unit late (a gap seen at the next unit's first packet
    still counts for it) and only whole ones (see the module docstring)."""

    def __init__(self, out: BinaryIO, damage: _Damage | None):
        self.out = out
        self.damage = damage
        self.written = 0
        self.dropped: list[dict] = []
        self.dropped_count = 0
        self._pending: h265_rtp.RtpAccessUnit | None = None
        self._index = 0

    def add(self, access_unit: h265_rtp.RtpAccessUnit) -> None:
        if self.damage is None:
            self.out.write(h265_rtp.to_annexb(access_unit.nals))
            self.written += 1
            return
        self.flush()
        self._pending = access_unit

    def flush(self) -> None:
        unit, self._pending = self._pending, None
        if unit is None:
            return
        index, self._index = self._index, self._index + 1
        reasons = []
        if unit.timestamp in self.damage.timestamps:
            reasons.append("lost or late packet")
        if not unit.marker:
            reasons.append("no marker")
        if not any(h265_rtp.is_vcl(h265_rtp.nal_type(nal)) for nal in unit.nals):
            reasons.append("no slice")
        if not reasons:
            self.out.write(h265_rtp.to_annexb(unit.nals))
            self.written += 1
            return
        self.out.write(h265_rtp.to_annexb(n for n in unit.nals if h265_rtp.nal_type(n) in PARAMETER_SET_NAMES))
        self.dropped_count += 1
        if len(self.dropped) < DROPPED_LISTED_MAX:
            self.dropped.append({"access_unit": index, "timestamp": unit.timestamp, "why": reasons})

    def summary(self) -> dict:
        return {
            "drop_incomplete_aus": self.damage is not None,
            "access_units_written": self.written,
            "access_units_dropped": self.dropped_count,
            "dropped": self.dropped,
        }


def analyze_rtpcap(
    path: str, *, donl: bool = False, h265_out: BinaryIO | None = None, drop_incomplete_aus: bool = False
) -> dict:
    """Packet statistics, depacketizer counters and bitstream summary of a .rtpcap file."""
    packets = rtp_capture.PacketStats()
    depacketizer = h265_rtp.Depacketizer(donl=donl)
    units = h265_rtp.RtpAccessUnits()
    bitstream = BitstreamStats()
    damage = _Damage() if drop_incomplete_aus else None
    writer = _AnnexBWriter(h265_out, damage) if h265_out is not None else None
    markerless = 0

    def take(access_unit: h265_rtp.RtpAccessUnit) -> None:
        nonlocal markerless
        markerless += not access_unit.marker
        bitstream.add_access_unit(access_unit.nals)
        if writer is not None:
            writer.add(access_unit)

    with rtpcap.Reader(path) as reader:
        meta = reader.meta
        digest = rtpcap.packets_sha256(_record(reader, packets, depacketizer, units, take, damage))
    depacketizer.flush()
    last = units.flush()
    if last is not None:
        take(last)
    packet_summary = packets.summary()
    info = {
        "input": path,
        "format": "rtpcap",
        "meta": meta,
        "packets_sha256": digest,
        "rtp": packet_summary,
        "depacketizer": {
            "packets": dict(sorted(depacketizer.packets.items())),
            "fu_dropped": depacketizer.fu_dropped,
            "access_units_without_marker": markerless,
        },
        "stream": bitstream.summary(packet_summary["fps_from_timestamps"]),
    }
    if writer is not None:
        writer.flush()
        info["annexb"] = writer.summary()
    return info


def _record(reader, packets, depacketizer, units, take, damage: _Damage | None = None) -> Iterable[bytes]:
    """Feed every packet of `reader` to the statistics and the depacketizer, yielding it for the digest."""
    for time_ns, packet in reader:
        packets.add(time_ns, packet)
        try:
            header = h265_rtp.parse_rtp_header(packet)
        except h265_rtp.RtpError:
            yield packet
            continue
        nals = depacketizer.feed(header, packet[header.payload_offset : header.payload_end])
        if damage is not None:
            damage.packet(header, depacketizer.fu_dropped)
        for access_unit in units.add(header, nals, time_ns):
            take(access_unit)
        yield packet


def analyze_annexb(path: str, *, fps: float | None, h265_out: BinaryIO | None = None) -> dict:
    """Bitstream summary of an Annex-B file. The frame rate comes from --fps, else from the SPS VUI timing info."""
    bitstream = BitstreamStats()
    units = h265_rtp.AnnexBAccessUnits()
    with open(path, "rb") as stream:
        for nal in h265_rtp.iter_annexb(stream):
            done = units.add(nal)
            if done is not None:
                bitstream.add_access_unit(done)
            if h265_out is not None:
                h265_out.write(h265_rtp.to_annexb((nal,)))
    last = units.flush()
    if last is not None:
        bitstream.add_access_unit(last)
    if fps is None:
        vui_fps = [sps.vui.fps for sps in bitstream.sps.values() if sps.vui is not None and sps.vui.fps]
        fps = vui_fps[0] if vui_fps else None
    return {"input": path, "format": "annexb", "stream": bitstream.summary(fps)}


def print_summary(info: dict) -> None:
    print(f"{info['input']} ({info['format']})")
    if "rtp" in info:
        rtp = info["rtp"]
        print(
            f"  packets {rtp['packets']}  bytes {rtp['bytes']}  over {rtp['duration_s']} s  "
            f"{rtp['packets_per_s']} pkt/s  {rtp['bitrate_mbps']} Mbit/s"
        )
        print(
            f"  access units {rtp['access_units']}  markers {rtp['markers']}  RTP timestamp step "
            f"{rtp['timestamp_step_median']} -> {rtp['fps_from_timestamps']} fps (send/arrival {rtp['fps_from_arrival']})"
        )
        print(
            f"  seq missing {rtp['seq_missing']}  out of order {rtp['seq_out_of_order']}  wraps {rtp['seq_wraps']}  "
            f"ssrc {rtp['ssrcs']}  pt {rtp['payload_types']}"
        )
        print(f"  packet bytes {rtp['packet_bytes']}  packetization {rtp['packetization_share']}")
        print(f"  access unit bytes {rtp['au_bytes']}  span ms {rtp['au_span_ms']}")
        print(f"  depacketizer {info['depacketizer']}  packets_sha256 {info['packets_sha256']}")
    if "annexb" in info:
        a = info["annexb"]
        print(
            f"  Annex-B: {a['access_units_written']} access units written, {a['access_units_dropped']} left out "
            f"(--drop-incomplete-aus {a['drop_incomplete_aus']}); first left out: {a['dropped'][:5]}"
        )
    s = info["stream"]
    if s.get("profile") is None:
        print("  no SPS/PPS seen")
    else:
        print(
            f"  {s['profile']} profile, {s['tier']} tier, level {s['level']}; {s['width']}x{s['height']} "
            f"(coded {s['coded_width']}x{s['coded_height']}), {s['chroma_format']}, {s['bit_depth_luma']} bit; "
            f"CTB {s['ctb_size']} ({s['ctbs_per_row']}x{s['ctb_rows']}), SAO {s['sao']}; VUI {s['vui']}"
        )
        print(
            f"  WPP {s['wpp']}  tiles {s['tiles']}  dependent slices enabled {s['dependent_slice_segments_enabled']}  "
            f"deblocking disabled in PPS {s['deblocking_disabled_in_pps']}  weighted pred {s['weighted_pred']}"
        )
    print(
        f"  access units {s['access_units']}  fps {s['fps']}  duration {s['duration_s']} s  "
        f"ES {s['es_bitrate_mbps']} Mbit/s"
    )
    print(
        f"  slice segments per AU {s['slice_segments_per_au']}  dependent {s['dependent_slice_segments']}  "
        f"slice types {s['slice_types']}  reordered pictures {s['reordered_pictures']}  "
        f"sps_max_num_reorder_pics {s.get('sps_max_num_reorder_pics')}"
    )
    print(
        f"  IRAP {s['irap_count']} (IDR {s['idr_count']}, CRA {s['cra_count']}), first at AU {s['first_irap_au']}, "
        f"interval frames {s['irap_interval_frames']} s {s['irap_interval_s']}"
    )
    print(f"  parameter sets {s['parameter_sets']}")
    print(f"  IRAP AU bytes {s['irap_au_bytes']}  other AU bytes {s['other_au_bytes']}")
    if s["errors"]:
        print(f"  errors {s['errors']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help=".rtpcap file, or an Annex-B H.265 file")
    parser.add_argument("--json", default=None, help="write the full summary as JSON to this path ('-' = stdout)")
    parser.add_argument("--write-h265", default=None, help="write the stream as Annex-B to this path")
    parser.add_argument(
        "--drop-incomplete-aus",
        action="store_true",
        help="with --write-h265 and a .rtpcap: leave out access units a lost or late packet may have cut short",
    )
    parser.add_argument("--fps", type=float, default=None, help="frame rate of an Annex-B input without VUI timing")
    parser.add_argument("--donl", action="store_true", help="RTP payloads carry DONL fields (sprop-max-don-diff > 0)")
    args = parser.parse_args(argv)

    with open(args.input, "rb") as probe:
        is_rtpcap = probe.read(len(rtpcap.MAGIC)) == rtpcap.MAGIC
    if args.drop_incomplete_aus and not (is_rtpcap and args.write_h265):
        parser.error("--drop-incomplete-aus needs a .rtpcap input and --write-h265")
    h265_out = open(args.write_h265, "wb") if args.write_h265 else None  # noqa: SIM115
    try:
        if is_rtpcap:
            info = analyze_rtpcap(
                args.input, donl=args.donl, h265_out=h265_out, drop_incomplete_aus=args.drop_incomplete_aus
            )
        else:
            info = analyze_annexb(args.input, fps=args.fps, h265_out=h265_out)
    finally:
        if h265_out is not None:
            h265_out.close()
    if args.json != "-":
        print_summary(info)
    if args.json:
        text = json.dumps(info, indent=2) + "\n"
        if args.json == "-":
            sys.stdout.write(text)
        else:
            with open(args.json, "w") as out:
                out.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
