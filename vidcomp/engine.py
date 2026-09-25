"""Core engine: probing, content analysis, planning and encoding.

No GUI dependencies - used by both the Qt app and the CLI.

How it picks resolution / framerate
-----------------------------------
1. A quick *complexity probe* encodes a few 1 s samples of the source at a fixed
   quantizer (constant QP) on the GPU. The bits that costs tells us how "hard" the
   content is (static screen recording vs. fast gameplay with foliage).
2. From that we model how many kbps the content needs to look good at any
   resolution / framerate:  needed = B_src * (pixels ratio)^0.75 * (fps ratio)^0.7
3. Given the bitrate budget from the target size, we search every allowed fps and
   the best resolution for it, minimising a weighted loss:
        w_res * (resolution drop)^2 + w_fps * (fps drop)^2 + w_q * (quality shortfall)^2
   The "priority" slider moves weight between w_res and w_fps, so you can keep
   60 fps and drop resolution, keep 1440p and drop fps, or split the difference.
4. Encode with a hardware encoder (NVENC etc.), check the result size and
   re-encode with a corrected bitrate if it overshot (cheap, because it's fast).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, asdict
from fractions import Fraction
from typing import Callable

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

_WIN = os.name == "nt"
_CREATE_NO_WINDOW = 0x08000000
_BELOW_NORMAL_PRIORITY = 0x00004000

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts",
              ".mts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".gif"}

STANDARD_HEIGHTS = [2160, 1800, 1440, 1296, 1200, 1080, 1008, 900, 864, 810, 720,
                    648, 576, 540, 480, 432, 360, 288, 240, 144]

# Reference point for the complexity probe / bitrate model.
PROBE_QP = 30
DEFAULT_BPP = 0.03          # bits/pixel/frame at PROBE_QP with hevc_nvenc when not probed
RES_EXP = 0.75              # bits needed ~ pixels^RES_EXP
FPS_EXP = 0.70              # bits needed ~ fps^FPS_EXP
QP_PER_OCTAVE = 6.0         # +6 QP ~= half the bits


class EncodeError(Exception):
    pass


class Cancelled(Exception):
    pass


def _popen_kwargs(low_priority: bool = False) -> dict:
    if not _WIN:
        return {}
    flags = _CREATE_NO_WINDOW
    if low_priority:
        flags |= _BELOW_NORMAL_PRIORITY
    return {"creationflags": flags}


# --------------------------------------------------------------------------- encoders

@dataclass(frozen=True)
class EncoderSpec:
    name: str           # ffmpeg encoder name
    label: str
    codec: str          # h264 | hevc | av1
    family: str         # nvenc | qsv | amf | cpu
    efficiency: float   # bits needed relative to hevc_nvenc (lower = better)
    ten_bit: bool       # can do 10-bit output


ENCODERS: list[EncoderSpec] = [
    EncoderSpec("h264_nvenc", "H.264 - NVIDIA NVENC  (plays everywhere)", "h264", "nvenc", 1.35, False),
    EncoderSpec("hevc_nvenc", "H.265 / HEVC - NVIDIA NVENC  (recommended)", "hevc", "nvenc", 1.00, True),
    EncoderSpec("av1_nvenc", "AV1 - NVIDIA NVENC  (best quality per MB)", "av1", "nvenc", 0.82, True),
    EncoderSpec("h264_qsv", "H.264 - Intel QuickSync", "h264", "qsv", 1.40, False),
    EncoderSpec("hevc_qsv", "HEVC - Intel QuickSync", "hevc", "qsv", 1.05, True),
    EncoderSpec("av1_qsv", "AV1 - Intel QuickSync", "av1", "qsv", 0.88, True),
    EncoderSpec("h264_amf", "H.264 - AMD AMF", "h264", "amf", 1.50, False),
    EncoderSpec("hevc_amf", "HEVC - AMD AMF", "hevc", "amf", 1.10, True),
    EncoderSpec("av1_amf", "AV1 - AMD AMF", "av1", "amf", 0.90, True),
    EncoderSpec("libx264", "H.264 - CPU x264  (slow, 2-pass)", "h264", "cpu", 1.10, False),
    EncoderSpec("libx265", "HEVC - CPU x265  (very slow, 2-pass)", "hevc", "cpu", 0.85, True),
    EncoderSpec("libsvtav1", "AV1 - CPU SVT-AV1  (slow)", "av1", "cpu", 0.75, True),
]
ENCODER_BY_NAME = {e.name: e for e in ENCODERS}

_available_cache: list[EncoderSpec] | None = None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def available_encoders(refresh: bool = False) -> list[EncoderSpec]:
    """Encoders compiled into ffmpeg AND actually usable on this machine (hardware
    encoders are test-encoded, since being compiled in doesn't mean the GPU exists)."""
    global _available_cache
    if _available_cache is not None and not refresh:
        return _available_cache
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20, **_popen_kwargs()).stdout
    except Exception:
        _available_cache = []
        return []
    compiled = [e for e in ENCODERS if re.search(rf"\s{re.escape(e.name)}\s", out)]

    def works(e: EncoderSpec) -> bool:
        if e.family == "cpu":
            return True
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
               "-i", "color=black:s=320x240:r=30:d=0.2", "-c:v", e.name, "-f", "null", "-"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=20,
                                  **_popen_kwargs()).returncode == 0
        except Exception:
            return False

    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(works, compiled))
    _available_cache = [e for e, good in zip(compiled, ok) if good]
    return _available_cache


def nvenc_load() -> tuple[int, int] | None:
    """(active encoder sessions, encoder utilisation %) from nvidia-smi, or None.
    Used to warn when another app (OBS replay buffer, ShadowPlay...) is sharing NVENC."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=encoder.stats.sessionCount,utilization.encoder",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5, **_popen_kwargs()).stdout
        sessions, util = (int(x) for x in out.splitlines()[0].split(","))
        return sessions, util
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def cuda_available() -> bool:
    return any(e.family == "nvenc" for e in available_encoders())


# --------------------------------------------------------------------------- probing

@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int
    bitrate: int | None  # bps


@dataclass
class MediaInfo:
    path: str
    size: int
    duration: float
    width: int            # display width (after rotation)
    height: int           # display height (after rotation)
    fps: float
    fps_frac: str
    vcodec: str
    pix_fmt: str
    bit_depth: int
    rotation: int
    hdr: bool
    color_trc: str
    color_primaries: str
    color_space: str
    vbitrate: int | None
    audio: list[AudioStream] = field(default_factory=list)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    def describe(self) -> str:
        return f"{self.short_side}p{fmt_fps(self.fps)} · {fmt_dur(self.duration)} · {fmt_size(self.size)}"


def _frac(s: str | None) -> Fraction | None:
    try:
        if not s or s in ("0/0", "0"):
            return None
        f = Fraction(s)
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: str) -> MediaInfo:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=60, **_popen_kwargs())
    if p.returncode != 0:
        raise EncodeError(p.stderr.strip() or "ffprobe failed")
    data = json.loads(p.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise EncodeError("No video stream found")
    v = vids[0]

    rate = _frac(v.get("avg_frame_rate"))
    rrate = _frac(v.get("r_frame_rate"))
    if rate is None or rate > 480:
        rate = rrate
    if rate is None:
        rate = Fraction(30)

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    if not rotation and "rotate" in v.get("tags", {}):
        try:
            rotation = int(v["tags"]["rotate"])
        except ValueError:
            pass
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    if abs(rotation) % 180 == 90:
        w, h = h, w

    duration = 0.0
    for src in (fmt.get("duration"), v.get("duration")):
        try:
            duration = float(src)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise EncodeError("Could not determine duration")

    pix_fmt = v.get("pix_fmt", "") or ""
    bit_depth = 10 if re.search(r"(10|12)(le|be)?$", pix_fmt) or "p010" in pix_fmt else 8
    trc = v.get("color_transfer", "") or ""
    hdr = trc in ("smpte2084", "arib-std-b67")

    def _br(d: dict) -> int | None:
        for key in ("bit_rate",):
            try:
                return int(d[key])
            except (KeyError, ValueError, TypeError):
                pass
        for k, val in (d.get("tags") or {}).items():
            if k.upper().startswith("BPS"):
                try:
                    return int(val)
                except ValueError:
                    pass
        return None

    audio = [AudioStream(i, a.get("codec_name", "?"), int(a.get("channels", 2) or 2), _br(a))
             for i, a in enumerate(s for s in streams if s.get("codec_type") == "audio")]

    size = int(fmt.get("size") or os.path.getsize(path))
    return MediaInfo(
        path=path, size=size, duration=duration, width=w, height=h,
        fps=float(rate), fps_frac=f"{rate.numerator}/{rate.denominator}",
        vcodec=v.get("codec_name", "?"), pix_fmt=pix_fmt, bit_depth=bit_depth,
        rotation=rotation, hdr=hdr, color_trc=trc,
        color_primaries=v.get("color_primaries", "") or "",
        color_space=v.get("color_space", "") or "",
        vbitrate=_br(v), audio=audio)


def analyze_complexity(info: MediaInfo, trim: tuple[float | None, float | None] = (None, None),
                       samples: int = 5, seg_len: float = 1.0) -> float | None:
    """Encode a few short samples at constant QP and return bits/pixel/frame.

    Uses hevc_nvenc when available (very fast), otherwise libx264 ultrafast (scaled to
    match). Returns None if probing fails - the planner then falls back to a default.
    """
    start, end = _trim_bounds(info, trim)
    span = end - start
    if span <= 0:
        return None
    encs = {e.name for e in available_encoders()}
    if "hevc_nvenc" in encs:
        venc = ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", str(PROBE_QP)]
        scale = 1.0
    else:
        venc = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", str(PROBE_QP), "-threads", "4"]
        scale = 0.6

    if span < samples * seg_len * 3:
        points = [(start, min(span, 6.0))]
    else:
        points = [(start + span * (0.08 + 0.84 * k / (samples - 1)), seg_len) for k in range(samples)]

    def one(pt):
        ss, dur = pt
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-threads", "4", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
               "-i", info.path, "-map", "0:v:0", "-an", "-vf", "format=yuv420p", *venc, "-f", "null", "-"]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=120, **_popen_kwargs(True))
        frames = re.findall(r"frame=\s*(\d+)", p.stderr)
        vk = re.search(r"video:\s*([\d.]+)\s*(K|k|M)i?B", p.stderr)
        if not frames or not vk:
            return None
        kb = float(vk.group(1)) * (1024 if vk.group(2) == "M" else 1)
        return int(frames[-1]), kb * 1024 * 8

    try:
        with ThreadPoolExecutor(len(points)) as ex:
            res = [r for r in ex.map(one, points) if r and r[0] > 0]
    except Exception:
        return None
    if not res:
        return None
    frames = sum(r[0] for r in res)
    bits = sum(r[1] for r in res)
    return bits / frames / (info.width * info.height) * scale


# --------------------------------------------------------------------------- settings

@dataclass
class Settings:
    # target
    target_mode: str = "size"          # size | percent | bitrate | quality
    target_size: float = 10.0
    size_unit: str = "MB"              # MB (1000^2) | MiB (1024^2)
    target_percent: float = 25.0
    target_bitrate: int = 3000         # total kbps (video + audio)
    cq: int = 28                       # constant quality value (quality mode)
    safety_margin: float = 3.0         # % below target to aim for
    max_attempts: int = 3              # re-encodes allowed if the result overshoots
    if_smaller: str = "copy"           # copy | skip | encode  (source already under target)

    # resolution  (all heights refer to the SHORT side, so 1080p portrait works too)
    res_mode: str = "auto"             # auto | source | fixed
    fixed_height: int = 1080
    min_height: int = 360
    max_height: int = 2160
    snap_standard: bool = False

    # framerate
    fps_mode: str = "auto"             # auto | source | fixed
    fixed_fps: float = 30.0
    min_fps: float = 24.0
    max_fps: float = 240.0
    fps_ladder: str = "60, 50, 48, 30, 25, 24, 20, 15"
    prefer_clean_fps: bool = True      # prefer fps that divide the source evenly (no judder)

    # trade-offs
    priority: int = 50                 # 0 = keep smoothness (fps)  ...  100 = keep detail (resolution)
    quality_bias: int = 50             # 0 = cleaner frames (downscale more) ... 100 = max resolution
    smart_probe: bool = True

    # video encoder
    encoder: str = "hevc_nvenc"
    speed: int = 4                     # 1 fastest ... 7 best quality (NVENC p4: p5+ is slower, no better)
    rate_control: str = "vbr"          # vbr | cbr
    multipass: str = "disabled"        # disabled | qres | fullres  (NVENC; measured: no VMAF gain, ~3% slower)
    ten_bit: bool = False
    spatial_aq: bool = True
    temporal_aq: bool = True
    lookahead: int = 0                 # measured: no VMAF gain, ~8% slower
    decoder: str = "auto"              # auto | cpu | gpu
    tonemap_hdr: bool = True
    scaler: str = "lanczos"            # lanczos | bicubic | bilinear

    # audio
    audio_mode: str = "first"          # first | mix | all | none
    audio_codec: str = "aac"           # aac | opus | copy
    audio_bitrate: int = 128
    audio_auto: bool = True            # lower audio bitrate when the budget is tiny
    audio_mono: bool = False

    # output
    output_dir: str = ""               # empty = next to the source
    name_template: str = "{name}_{target}"
    container: str = "mp4"             # mp4 | mkv
    overwrite: bool = False
    strip_metadata: bool = False

    # performance
    parallel_jobs: int = 2
    low_priority: bool = True
    turbo: bool = False                # trade a little quality for ~1.5-1.7x speed (see apply_turbo)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        valid = {f.name: f for f in fields(cls)}
        for k, v in (d or {}).items():
            if k in valid:
                try:
                    default = getattr(s, k)
                    setattr(s, k, type(default)(v) if default is not None else v)
                except (TypeError, ValueError):
                    pass
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    def target_bytes(self, info: MediaInfo | None = None, duration: float | None = None) -> int | None:
        if self.target_mode == "size":
            return int(self.target_size * (1_000_000 if self.size_unit == "MB" else 1_048_576))
        if self.target_mode == "percent" and info is not None:
            frac = (duration / info.duration) if duration else 1.0
            return int(info.size * self.target_percent / 100 * frac)
        return None

    def target_label(self) -> str:
        if self.target_mode == "size":
            return f"{self.target_size:g}{self.size_unit}"
        if self.target_mode == "percent":
            return f"{self.target_percent:g}pct"
        if self.target_mode == "bitrate":
            return f"{self.target_bitrate}k"
        return f"cq{self.cq}"


# --------------------------------------------------------------------------- planning

@dataclass
class Plan:
    action: str                 # encode | copy | skip
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_filter: str | None = None     # value for fps= filter, None = keep source
    video_kbps: float | None = None   # None in quality mode
    audio_kbps: int = 0               # per stream
    audio_streams: int = 0
    target_bytes: int | None = None
    duration: float = 0.0
    quality: str = ""                 # Excellent / Good / OK / Low / Very low
    est_qp: float | None = None
    complexity: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    def est_bytes(self) -> int | None:
        if self.target_bytes:
            return self.target_bytes
        if self.video_kbps is None:
            return None
        return int((self.video_kbps + self.audio_kbps * self.audio_streams) * 1000 / 8 * self.duration)

    def summary(self) -> str:
        if self.action == "skip":
            return "skip - already small enough"
        if self.action == "copy":
            return "copy - already small enough"
        s = f"{self.short_side}p{fmt_fps(self.fps)}"
        if self.video_kbps is not None:
            s += f" · {fmt_kbps(self.video_kbps)}"
        return s


def _trim_bounds(info: MediaInfo, trim) -> tuple[float, float]:
    start = max(0.0, trim[0] or 0.0) if trim else 0.0
    end = min(info.duration, trim[1]) if trim and trim[1] else info.duration
    return start, max(start, end)


def parse_ladder(text: str) -> list[float]:
    out = []
    for part in re.split(r"[,\s;]+", text or ""):
        try:
            v = float(part)
            if 1 <= v <= 1000:
                out.append(v)
        except ValueError:
            pass
    return out


def _fps_candidates(info: MediaInfo, s: Settings) -> list[tuple[float, str | None, bool]]:
    """Returns [(fps, fps_filter_value_or_None, clean_ratio)]."""
    src = Fraction(info.fps_frac)
    src_f = float(src)

    def resolve(target: float) -> tuple[float, str | None, bool]:
        if target >= src_f * 0.97:
            return src_f, None, True
        k = src_f / target
        if abs(k - round(k)) < 0.03 * k and round(k) >= 2:
            exact = src / round(k)
            return float(exact), f"{exact.numerator}/{exact.denominator}", True
        return target, f"{target:g}", False

    if s.fps_mode == "source":
        return [(src_f, None, True)]
    if s.fps_mode == "fixed":
        return [resolve(min(s.fixed_fps, src_f))]

    hi = min(src_f, s.max_fps)
    lo = min(s.min_fps, hi)
    cands = {resolve(src_f) if src_f <= s.max_fps * 1.03 else resolve(hi)}
    for v in parse_ladder(s.fps_ladder):
        if lo * 0.97 <= v <= hi * 1.03:
            cands.add(resolve(v))
    out: dict[str, tuple] = {}
    for c in cands:
        out.setdefault(f"{c[0]:.3f}", c)
    return sorted(out.values(), key=lambda c: -c[0])


def _round_dims(info: MediaInfo, short: float, snap: bool) -> tuple[int, int]:
    src_short = info.short_side
    if short >= src_short - 1:
        return info.width, info.height
    if snap:
        cands = [h for h in STANDARD_HEIGHTS if h <= short * 1.03]
        short = cands[0] if cands else short
    short = max(64, int(round(short / 8)) * 8)
    r = short / src_short
    w = max(2, int(round(info.width * r / 2)) * 2)
    h = max(2, int(round(info.height * r / 2)) * 2)
    if info.width <= info.height:
        w = short
    else:
        h = short
    return w, h


def quality_label(qp: float) -> str:
    if qp <= 25:
        return "Excellent"
    if qp <= 29.5:
        return "Good"
    if qp <= 33.5:
        return "OK"
    if qp <= 38:
        return "Low"
    return "Very low"


def make_plan(info: MediaInfo, s: Settings, complexity: float | None = None,
              trim: tuple[float | None, float | None] = (None, None)) -> Plan:
    start, end = _trim_bounds(info, trim)
    dur = max(0.1, end - start)
    trimmed = (start > 0.01) or (end < info.duration - 0.01)
    enc = ENCODER_BY_NAME.get(s.encoder, ENCODERS[0])
    plan = Plan(action="encode", duration=dur, complexity=complexity)

    # ---- audio
    n_audio = 0
    if info.audio and s.audio_mode != "none":
        n_audio = len(info.audio) if s.audio_mode == "all" else 1
    a_kbps = s.audio_bitrate
    if s.audio_codec == "copy":
        srcs = info.audio if s.audio_mode == "all" else info.audio[:1]
        a_kbps = max([int((a.bitrate or 192000) / 1000) for a in srcs] or [0])
        if s.audio_mode == "mix":  # mixing can't be stream-copied
            a_kbps = s.audio_bitrate

    # ---- budget
    tbytes = s.target_bytes(info, dur if trimmed else None)
    plan.target_bytes = tbytes
    budget = None  # total kbps
    if tbytes:
        if (not trimmed and info.size <= tbytes and s.if_smaller != "encode"):
            plan.action = s.if_smaller
            plan.width, plan.height, plan.fps = info.width, info.height, info.fps
            return plan
        budget = tbytes * 8 / dur / 1000 * (1 - s.safety_margin / 100)
        budget -= 2.0  # container overhead, roughly
    elif s.target_mode == "bitrate":
        budget = float(s.target_bitrate)

    if budget is not None and n_audio:
        if s.audio_auto and s.audio_codec != "copy" and a_kbps * n_audio > budget * 0.12:
            steps = [24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 320]
            floor = 24 if s.audio_codec == "opus" else 48
            want = budget * 0.12 / n_audio
            a_kbps = max([x for x in steps if x <= want] or [floor])
            a_kbps = max(floor, min(a_kbps, s.audio_bitrate))
            plan.notes.append(f"audio lowered to {a_kbps}k to leave room for video")
    plan.audio_kbps, plan.audio_streams = a_kbps, n_audio

    video = None
    if budget is not None:
        video = budget - a_kbps * n_audio
        if video < 60:
            plan.notes.append("target is extremely small for this length - expect a mess")
            video = 60.0
        if info.vbitrate and video > info.vbitrate / 1000 * 1.05:
            video = info.vbitrate / 1000 * 1.05
    plan.video_kbps = video

    # ---- resolution / fps search
    src_short = info.short_side
    fps_cands = _fps_candidates(info, s)
    if s.res_mode == "source":
        short_lo = short_hi = src_short
    elif s.res_mode == "fixed":
        short_lo = short_hi = min(s.fixed_height, src_short)
    else:
        short_hi = min(src_short, s.max_height)
        short_lo = min(s.min_height, short_hi)

    qp_target = 24 + 12 * s.quality_bias / 100
    bpp = complexity if complexity else DEFAULT_BPP
    need_src = bpp * info.width * info.height * info.fps * enc.efficiency / 1000  # kbps @ PROBE_QP
    need_src *= 2 ** ((PROBE_QP - qp_target) / QP_PER_OCTAVE)                  # kbps @ qp_target

    if video is None:  # constant-quality mode: no budget, just honour locks / caps
        f, filt, _ = fps_cands[0]
        w, h = _round_dims(info, short_hi, s.snap_standard)
        plan.width, plan.height, plan.fps, plan.fps_filter = w, h, f, filt
        plan.quality = "CQ"
        return plan

    p = s.priority / 100
    w_res = math.exp(4 * (p - 0.5))
    w_fps = math.exp(-4 * (p - 0.5))
    w_q = 2.5
    k_res = 2 * RES_EXP  # octaves of bits per octave of short side

    s0 = math.log2(need_src / video)  # quality shortfall at source res/fps, in bit-octaves
    a_lo = math.log2(src_short / short_hi)
    a_hi = math.log2(src_short / short_lo)
    best = None
    for f, filt, clean in fps_cands:
        b = math.log2(info.fps / f) if f < info.fps else 0.0      # octaves of frames saved
        # perceived loss: halving above 60 fps (120->60) is far less visible than 60->30
        hi_src, lo_src = max(info.fps, 60.0), min(info.fps, 60.0)
        b_seen = math.log2(lo_src / min(f, lo_src)) + 0.25 * math.log2(hi_src / max(f, 60.0))
        rem = s0 - FPS_EXP * b
        a = k_res * w_q * rem / (w_res + k_res ** 2 * w_q) if rem > 0 else 0.0
        a = min(max(a, a_lo), a_hi)
        short = src_short / 2 ** a
        wd, ht = _round_dims(info, short, s.snap_standard)
        a = math.log2(src_short / min(wd, ht))
        short_fall = s0 - FPS_EXP * b - k_res * a
        loss = w_res * a * a + w_fps * b_seen * b_seen + w_q * max(0.0, short_fall) ** 2
        if s.prefer_clean_fps and not clean:
            loss += 0.35
        if best is None or loss < best[0]:
            best = (loss, wd, ht, f, filt, short_fall)

    _, wd, ht, f, filt, short_fall = best
    plan.width, plan.height, plan.fps, plan.fps_filter = wd, ht, f, filt
    plan.est_qp = qp_target + QP_PER_OCTAVE * short_fall
    plan.quality = quality_label(plan.est_qp)
    return plan


# --------------------------------------------------------------------------- command building

_SPEED = {
    "nvenc": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    "qsv": ["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
    "amf": ["speed", "speed", "balanced", "balanced", "quality", "quality", "quality"],
    "libx264": ["ultrafast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
    "libx265": ["ultrafast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
    "libsvtav1": ["12", "10", "8", "7", "6", "5", "4"],
}


def _use_gpu_decode(info: MediaInfo, s: Settings, enc: EncoderSpec) -> bool:
    if enc.family != "nvenc" or info.rotation % 360 != 0:
        return False
    if info.hdr and s.tonemap_hdr:
        return False
    if info.vcodec not in ("h264", "hevc", "av1", "vp9", "vp8", "mpeg2video", "mpeg4", "vc1"):
        return False
    if s.decoder == "gpu":
        return True
    if s.decoder == "cpu":
        return False
    # auto: CPU decode (all cores) wins for most sources; NVDEC wins on heavy 4K HEVC/AV1.
    # auto: NVDEC for HEVC/AV1/VP9 (heavy to decode on CPU, e.g. OBS HEVC recordings - measured ~5%
    # faster end-to-end and frees the CPU); CPU for H.264 etc. where all cores are just as fast.
    return info.vcodec in ("hevc", "av1", "vp9")


def _video_filters(info: MediaInfo, plan: Plan, s: Settings, enc: EncoderSpec,
                   gpu_decode: bool) -> str:
    chain: list[str] = []
    scaling = (plan.width, plan.height) != (info.width, info.height)
    ten = s.ten_bit and enc.ten_bit
    if plan.fps_filter:
        chain.append(f"fps={plan.fps_filter}")

    if gpu_decode:
        algo = {"lanczos": 4, "bicubic": 2, "bilinear": 1}.get(s.scaler, 4)
        size = f"{plan.width}:{plan.height}:" if scaling else ""
        # 10-bit output from NVENC uses -highbitdepth, so frames stay 8-bit nv12 here
        # unless the source itself is 10-bit and we're keeping it.
        fmt = "p010le" if (ten and info.bit_depth > 8) else "nv12"
        chain.append(f"scale_cuda={size}interp_algo={algo}:format={fmt}")
        return ",".join(chain)

    tonemap = info.hdr and s.tonemap_hdr
    cpu_fmt = "yuv420p10le" if (ten and info.bit_depth > 8 and not tonemap) else "yuv420p"
    if tonemap:
        tin = info.color_trc or "smpte2084"
        pin = info.color_primaries or "bt2020"
        mat = info.color_space or "bt2020nc"
        size = f"w={plan.width}:h={plan.height}:filter=lanczos:" if scaling else ""
        chain.append(f"zscale={size}tin={tin}:pin={pin}:min={mat}:t=linear:npl=100")
        chain += ["format=gbrpf32le", "zscale=p=bt709", "tonemap=tonemap=hable:desat=0",
                  "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p"]
        return ",".join(chain)

    if enc.family == "nvenc" and scaling:
        algo = {"lanczos": 4, "bicubic": 2, "bilinear": 1}.get(s.scaler, 4)
        up_fmt = "p010le" if cpu_fmt == "yuv420p10le" else "nv12"
        chain += [f"format={up_fmt}", "hwupload_cuda",
                  f"scale_cuda={plan.width}:{plan.height}:interp_algo={algo}"]
        return ",".join(chain)

    if scaling:
        chain.append(f"scale={plan.width}:{plan.height}:flags={s.scaler}")
    chain.append(f"format={cpu_fmt}")
    return ",".join(chain)


def _video_codec_args(info: MediaInfo, plan: Plan, s: Settings, enc: EncoderSpec,
                      vkbps: float | None, pass_no: int | None) -> list[str]:
    speed_i = min(max(s.speed, 1), 7) - 1
    ten = s.ten_bit and enc.ten_bit
    a: list[str] = ["-c:v", enc.name]
    v = int(vkbps) if vkbps else None

    if enc.family == "nvenc":
        a += ["-preset", _SPEED["nvenc"][speed_i], "-tune", "hq"]
        if v is None:
            a += ["-rc", "vbr", "-cq", str(s.cq), "-b:v", "0"]
        elif s.rate_control == "cbr":
            a += ["-rc", "cbr", "-b:v", f"{v}k", "-bufsize", f"{v * 2}k"]
        else:
            a += ["-rc", "vbr", "-b:v", f"{v}k", "-maxrate", f"{int(v * 1.5)}k", "-bufsize", f"{v * 2}k"]
        if s.multipass != "disabled":
            a += ["-multipass", s.multipass]
        if s.spatial_aq:
            a += ["-spatial-aq", "1"]
        if s.temporal_aq:
            a += ["-temporal-aq", "1"]
        if s.lookahead > 0:
            a += ["-rc-lookahead", str(s.lookahead)]
        if enc.codec == "h264":
            a += ["-profile:v", "high"]
        elif ten:
            if info.bit_depth <= 8:
                a += ["-highbitdepth", "1"]
            if enc.codec == "hevc":
                a += ["-profile:v", "main10"]
    elif enc.family == "qsv":
        a += ["-preset", _SPEED["qsv"][speed_i]]
        if v is None:
            a += ["-global_quality", str(s.cq)]
        else:
            a += ["-b:v", f"{v}k", "-maxrate", f"{int(v * 1.5)}k", "-bufsize", f"{v * 2}k"]
    elif enc.family == "amf":
        a += ["-quality", _SPEED["amf"][speed_i]]
        if v is None:
            a += ["-rc", "cqp", "-qp_i", str(s.cq), "-qp_p", str(s.cq)]
        else:
            a += ["-rc", "vbr_peak", "-b:v", f"{v}k", "-maxrate", f"{int(v * 1.5)}k"]
    elif enc.name == "libx264":
        a += ["-preset", _SPEED["libx264"][speed_i]]
        a += ["-crf", str(s.cq)] if v is None else ["-b:v", f"{v}k"]
        if pass_no:
            a += ["-pass", str(pass_no), "-passlogfile", "vc2pass"]
    elif enc.name == "libx265":
        a += ["-preset", _SPEED["libx265"][speed_i]]
        params = ["log-level=error"]
        if v is None:
            a += ["-crf", str(s.cq)]
        else:
            a += ["-b:v", f"{v}k"]
            if pass_no:
                params += [f"pass={pass_no}", "stats=vc2pass.log"]
        a += ["-x265-params", ":".join(params)]
        if ten:
            a += ["-profile:v", "main10"]
    elif enc.name == "libsvtav1":
        a += ["-preset", _SPEED["libsvtav1"][speed_i]]
        a += ["-crf", str(min(63, s.cq + 6))] if v is None else ["-b:v", f"{v}k"]
    return a


def _audio_args(info: MediaInfo, plan: Plan, s: Settings) -> tuple[list[str], list[str]]:
    """Returns (map+filter args, codec args)."""
    if not plan.audio_streams:
        return [], ["-an"]
    maps: list[str] = []
    codec = s.audio_codec
    if s.audio_mode == "mix" and len(info.audio) > 1:
        ins = "".join(f"[0:a:{i}]" for i in range(len(info.audio)))
        maps += ["-filter_complex", f"{ins}amix=inputs={len(info.audio)}:normalize=0:dropout_transition=0[aout]",
                 "-map", "[aout]"]
        if codec == "copy":
            codec = "aac"
    elif s.audio_mode == "all":
        maps += ["-map", "0:a"]
    else:
        maps += ["-map", "0:a:0"]

    if codec == "copy":
        return maps, ["-c:a", "copy"]
    ca = ["-c:a", "libopus" if codec == "opus" else "aac", "-b:a", f"{plan.audio_kbps}k"]
    if s.audio_mono:
        ca += ["-ac", "1"]
    elif codec == "opus":
        ca += ["-ac", "2"]  # libopus chokes on some multichannel layouts
    return maps, ca


def apply_turbo(s: Settings) -> Settings:
    """Turbo mode: fastest presets that still look decent. Measured on a 1440p60 OBS clip at
    the same bitrate (with OBS using 42% of NVENC): H.264 2.5x -> 4.4x for -2.7 VMAF,
    AV1 2.3x -> 4.1x for -0.6 VMAF, HEVC -> 4.1x."""
    if not s.turbo:
        return s
    t = Settings.from_dict(s.to_dict())
    enc = ENCODER_BY_NAME.get(s.encoder, ENCODERS[0])
    t.speed = min(s.speed, 1 if enc.codec == "av1" and enc.family == "nvenc" else 2)
    t.temporal_aq = False
    t.lookahead = 0
    t.multipass = "disabled"
    return t


def build_commands(info: MediaInfo, plan: Plan, s: Settings, out_path: str,
                   trim=(None, None), video_kbps: float | None = None,
                   gpu_decode: bool | None = None) -> list[list[str]]:
    s = apply_turbo(s)
    enc = ENCODER_BY_NAME.get(s.encoder, ENCODERS[0])
    if gpu_decode is None:
        gpu_decode = _use_gpu_decode(info, s, enc)
    vk = video_kbps if video_kbps is not None else plan.video_kbps
    start, end = _trim_bounds(info, trim)

    base = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-progress", "pipe:1", "-nostats"]
    inp: list[str] = []
    if gpu_decode:
        inp += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "8"]
    elif enc.family == "nvenc":
        inp += ["-init_hw_device", "cuda=cu", "-filter_hw_device", "cu"]
    if start > 0.01:
        inp += ["-ss", f"{start:.3f}"]
    if end < info.duration - 0.01:
        inp += ["-t", f"{end - start:.3f}"]
    inp += ["-i", info.path]

    vf = _video_filters(info, plan, s, enc, gpu_decode)
    common_v = ["-map", "0:v:0"] + (["-vf", vf] if vf else [])
    color = []
    if info.hdr and s.tonemap_hdr and not gpu_decode:
        color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    amap, acodec = _audio_args(info, plan, s)
    meta = ["-map_metadata", "-1", "-map_chapters", "-1"] if s.strip_metadata else []
    mux = ["-movflags", "+faststart"] if s.container == "mp4" else []
    tail = ["-sn", "-dn", "-max_muxing_queue_size", "4096"]

    two_pass = enc.name in ("libx264", "libx265") and vk is not None
    if two_pass:
        p1 = base + inp + common_v + color + _video_codec_args(info, plan, s, enc, vk, 1) + \
            ["-an", "-sn", "-dn", "-f", "null", "NUL" if _WIN else "/dev/null"]
        p2 = base + inp + common_v + amap + color + _video_codec_args(info, plan, s, enc, vk, 2) + \
            acodec + meta + mux + tail + [out_path]
        return [p1, p2]
    return [base + inp + common_v + amap + color + _video_codec_args(info, plan, s, enc, vk, None) +
            acodec + meta + mux + tail + [out_path]]


# --------------------------------------------------------------------------- running

class CancelToken:
    def __init__(self):
        self._ev = threading.Event()
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def cancel(self):
        self._ev.set()
        with self._lock:
            for p in list(self._procs):
                try:
                    p.kill()
                except Exception:
                    pass

    def _add(self, p):
        with self._lock:
            self._procs.add(p)
        if self.cancelled:
            p.kill()

    def _remove(self, p):
        with self._lock:
            self._procs.discard(p)


ProgressCb = Callable[[float, float | None, str], None]  # (fraction, speed_x, stage)


def run_ffmpeg(cmd: list[str], duration: float, cb: ProgressCb | None, token: CancelToken | None,
               low_priority: bool, cwd: str | None, stage: str) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            cwd=cwd, **_popen_kwargs(low_priority))
    if token:
        token._add(proc)
    err_lines: list[str] = []

    def drain():
        for line in proc.stderr:
            err_lines.append(line.decode("utf-8", "replace").rstrip())
            del err_lines[:-40]
    t = threading.Thread(target=drain, daemon=True)
    t.start()
    t0 = time.monotonic()
    out_t = 0.0
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            key, _, val = line.partition("=")
            if key in ("out_time_us", "out_time_ms"):
                try:
                    out_t = int(val) / 1_000_000
                except ValueError:
                    continue
            elif key == "progress" and cb:
                wall = time.monotonic() - t0
                speed = out_t / wall if wall > 0.5 and out_t > 0 else None
                cb(min(1.0, out_t / duration) if duration else 0.0, speed, stage)
        proc.wait()
        t.join(timeout=2)
    finally:
        if token:
            token._remove(proc)
    if token and token.cancelled:
        raise Cancelled()
    if proc.returncode != 0:
        msg = "\n".join(l for l in err_lines if l.strip())[-1500:]
        raise EncodeError(msg or f"ffmpeg exited with code {proc.returncode}")


# Rate-control calibration: how much each encoder over/undershoots the requested
# bitrate. Learned from finished encodes so the first attempt usually lands on target.
# Seeds measured on real OBS clips; turbo presets overshoot a bit more.
_calib: dict[str, float] = {"hevc_nvenc|vbr": 1.04, "av1_nvenc|vbr": 1.04,
                            "h264_nvenc|vbr|turbo": 1.07, "hevc_nvenc|vbr|turbo": 1.08,
                            "av1_nvenc|vbr|turbo": 1.08}
_calib_lock = threading.Lock()


def calibration() -> dict[str, float]:
    with _calib_lock:
        return dict(_calib)


def set_calibration(d: dict) -> None:
    with _calib_lock:
        for k, v in (d or {}).items():
            try:
                _calib[str(k)] = min(1.4, max(0.75, float(v)))
            except (TypeError, ValueError):
                pass


def _calib_key(s: Settings) -> str:
    return f"{s.encoder}|{s.rate_control}" + ("|turbo" if s.turbo else "")


def _learn(s: Settings, ratio: float) -> None:
    key = _calib_key(s)
    with _calib_lock:
        old = _calib.get(key, 1.0)
        _calib[key] = min(1.4, max(0.75, old * 0.5 + old * ratio * 0.5))


@dataclass
class EncodeResult:
    output: str
    size: int
    attempts: int
    elapsed: float
    speed: float
    over_target: bool = False
    gpu_fallback: bool = False


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{stem} ({i}){ext}"):
        i += 1
    return f"{stem} ({i}){ext}"


def output_path_for(info: MediaInfo, plan: Plan, s: Settings, reserve: set[str] | None = None) -> str:
    src_dir, base = os.path.split(info.path)
    stem = os.path.splitext(base)[0]
    ext = "." + (s.container if plan.action == "encode" else os.path.splitext(base)[1].lstrip(".") or s.container)
    enc = ENCODER_BY_NAME.get(s.encoder, ENCODERS[0])
    tokens = {"name": stem, "target": s.target_label(), "res": f"{plan.short_side}p",
              "fps": fmt_fps(plan.fps).strip() or "", "codec": enc.codec}
    try:
        name = (s.name_template or "{name}_{target}").format(**tokens)
    except (KeyError, IndexError, ValueError):
        name = f"{stem}_{s.target_label()}"
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip() or stem
    out_dir = s.output_dir or src_dir
    path = os.path.join(out_dir, name + ext)
    if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(info.path)):
        path = os.path.join(out_dir, name + "_compressed" + ext)
    if not s.overwrite:
        path = unique_path(path)
        if reserve is not None:
            stem2, ext2 = os.path.splitext(path)
            i = 2
            while path in reserve:
                path = f"{stem2} ({i}){ext2}"
                i += 1
    if reserve is not None:
        reserve.add(path)
    return path


def encode(info: MediaInfo, plan: Plan, s: Settings, out_path: str, trim=(None, None),
           cb: ProgressCb | None = None, token: CancelToken | None = None) -> EncodeResult:
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if plan.action in ("copy", "skip"):
        if plan.action == "copy":
            shutil.copy2(info.path, out_path)
        return EncodeResult(out_path if plan.action == "copy" else "", info.size, 0, 0.0, 0.0)

    s = apply_turbo(s)
    enc = ENCODER_BY_NAME.get(s.encoder, ENCODERS[0])
    stem, ext = os.path.splitext(out_path)
    tmp_out = f"{stem}.part{ext}"
    work = tempfile.mkdtemp(prefix="vidcomp_")
    gpu = _use_gpu_decode(info, s, enc)
    gpu_fallback = False
    vk = plan.video_kbps
    if vk is not None and plan.target_bytes:
        vk = vk / calibration().get(_calib_key(s), 1.0)
    attempts = max(1, s.max_attempts) if plan.target_bytes else 1
    size = 0
    attempt = 0
    try:
        for attempt in range(1, attempts + 1):
            while True:
                cmds = build_commands(info, plan, s, tmp_out, trim, vk, gpu)
                try:
                    for i, cmd in enumerate(cmds):
                        stage = "Encoding"
                        if len(cmds) > 1:
                            stage = f"Pass {i + 1}/{len(cmds)}"
                        if attempt > 1:
                            stage += f" (retry {attempt}/{attempts})"
                        n = len(cmds)
                        sub = (lambda f, sp, st, i=i, n=n: cb((i + f) / n, sp, st)) if cb else None
                        run_ffmpeg(cmd, plan.duration, sub, token, s.low_priority, work, stage)
                    break
                except EncodeError:
                    if gpu:  # NVDEC can't handle this file - redo on CPU decode
                        gpu, gpu_fallback = False, True
                        continue
                    raise
            size = os.path.getsize(tmp_out)
            audio_bytes = plan.audio_kbps * plan.audio_streams * 1000 / 8 * plan.duration
            if vk is not None and plan.duration > 3:
                asked = vk * 1000 / 8 * plan.duration
                _learn(s, max(1.0, size - audio_bytes) / asked)
            if not plan.target_bytes or size <= plan.target_bytes or vk is None:
                break
            if attempt == attempts:
                break
            want = plan.target_bytes * (1 - s.safety_margin / 100) - audio_bytes
            got = max(1.0, size - audio_bytes)
            vk = max(30.0, vk * (want / got) * 0.97)
        os.replace(tmp_out, out_path)
    except BaseException:
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

    elapsed = time.monotonic() - t0
    return EncodeResult(out_path, size, attempt, elapsed,
                        plan.duration / elapsed if elapsed > 0 else 0.0,
                        over_target=bool(plan.target_bytes and size > plan.target_bytes),
                        gpu_fallback=gpu_fallback)


# --------------------------------------------------------------------------- formatting

def fmt_size(b: float | None) -> str:
    if b is None:
        return "?"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.0f} KB"


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_fps(f: float) -> str:
    if not f:
        return ""
    r = round(f)
    return str(r) if abs(f - r) < 0.05 else f"{f:.2f}"


def fmt_kbps(k: float) -> str:
    return f"{k / 1000:.1f} Mbps" if k >= 1000 else f"{k:.0f} kbps"


def parse_time(text: str) -> float | None:
    """'1:05', '65', '0:01:05.5' -> seconds. Empty -> None."""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Bad time: {text!r}")
    sec = 0.0
    for v in vals:
        sec = sec * 60 + v
    return sec
