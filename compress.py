"""Command-line video compressor.

Examples:
    python compress.py clip.mp4 -s 10
    python compress.py a.mp4 b.mkv -s 50 --lock-res --encoder av1_nvenc
    python compress.py folder\\ -s 25 --priority 80 --fps 30 -o out\\
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vidcomp import engine as E  # noqa: E402


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                out += [os.path.join(root, n) for n in sorted(names)
                        if os.path.splitext(n)[1].lower() in E.VIDEO_EXTS and ".part." not in n]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"not found: {p}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compress videos to a target size with GPU encoding.")
    ap.add_argument("inputs", nargs="+", help="files or folders")
    t = ap.add_argument_group("target")
    t.add_argument("-s", "--size", type=float, default=10, help="target size (default 10)")
    t.add_argument("--mib", action="store_true", help="size is in MiB instead of MB")
    t.add_argument("--percent", type=float, help="target %% of the original size instead")
    t.add_argument("--bitrate", type=int, help="total kbps instead of a size")
    t.add_argument("--cq", type=int, help="constant quality instead of a size")
    v = ap.add_argument_group("resolution / framerate")
    v.add_argument("--lock-res", action="store_true", help="keep source resolution")
    v.add_argument("--height", type=int, help="fixed output height (short side)")
    v.add_argument("--min-height", type=int, default=360)
    v.add_argument("--max-height", type=int, default=2160)
    v.add_argument("--lock-fps", action="store_true", help="keep source framerate")
    v.add_argument("--fps", type=float, help="fixed output framerate")
    v.add_argument("--min-fps", type=float, default=24)
    v.add_argument("--priority", type=int, default=50,
                   help="0 = keep fps, drop resolution ... 100 = keep resolution, drop fps")
    v.add_argument("--quality-bias", type=int, default=50,
                   help="0 = cleaner frames (downscale more) ... 100 = more pixels")
    v.add_argument("--no-probe", action="store_true", help="skip the content complexity analysis")
    e = ap.add_argument_group("encoding")
    e.add_argument("--encoder", default="h264_nvenc", choices=[x.name for x in E.ENCODERS])
    e.add_argument("--speed", type=int, default=5, help="1 fastest ... 7 best")
    e.add_argument("--ten-bit", action="store_true")
    e.add_argument("--audio", default="first", choices=["first", "mix", "all", "none"])
    e.add_argument("--audio-codec", default="aac", choices=["aac", "opus", "copy"])
    e.add_argument("--audio-bitrate", type=int, default=128)
    e.add_argument("--start", help="trim start (e.g. 1:05)")
    e.add_argument("--end", help="trim end")
    o = ap.add_argument_group("output")
    o.add_argument("-o", "--output-dir", default="")
    o.add_argument("--name", default="{name}_{target}", help="template: {name} {target} {res} {fps} {codec}")
    o.add_argument("-j", "--jobs", type=int, default=2, help="parallel encodes (default 2)")
    o.add_argument("--dry-run", action="store_true", help="only print the plan")
    a = ap.parse_args()

    s = E.Settings(
        target_size=a.size, size_unit="MiB" if a.mib else "MB",
        min_height=a.min_height, max_height=a.max_height, min_fps=a.min_fps,
        priority=a.priority, quality_bias=a.quality_bias, smart_probe=not a.no_probe,
        encoder=a.encoder, speed=a.speed, ten_bit=a.ten_bit,
        audio_mode=a.audio, audio_codec=a.audio_codec, audio_bitrate=a.audio_bitrate,
        output_dir=a.output_dir, name_template=a.name, parallel_jobs=a.jobs)
    if a.percent:
        s.target_mode, s.target_percent = "percent", a.percent
    elif a.bitrate:
        s.target_mode, s.target_bitrate = "bitrate", a.bitrate
    elif a.cq:
        s.target_mode, s.cq = "quality", a.cq
    if a.lock_res:
        s.res_mode = "source"
    elif a.height:
        s.res_mode, s.fixed_height = "fixed", a.height
    if a.lock_fps:
        s.fps_mode = "source"
    elif a.fps:
        s.fps_mode, s.fixed_fps = "fixed", a.fps
    trim = (E.parse_time(a.start), E.parse_time(a.end)) if (a.start or a.end) else (None, None)

    if not E.ffmpeg_available():
        sys.exit("ffmpeg/ffprobe not found on PATH")
    files = collect(a.inputs)
    if not files:
        sys.exit("no input files")
    if s.encoder not in {x.name for x in E.available_encoders()}:
        sys.exit(f"encoder {s.encoder} is not available on this machine")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    cfg = {}
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        E.set_calibration(cfg.get("calibration", {}))
    except (OSError, ValueError):
        pass

    lock = threading.Lock()
    reserved: set[str] = set()
    t0 = time.monotonic()

    def run(path):
        name = os.path.basename(path)
        try:
            info = E.probe(path)
            cx = E.analyze_complexity(info, trim) if s.smart_probe else None
            plan = E.make_plan(info, s, cx, trim)
            with lock:
                out = E.output_path_for(info, plan, s, reserved)
                print(f"{name}: {info.describe()}  ->  {plan.summary()} {plan.quality}")
            if a.dry_run:
                return
            last = [0.0]

            def cb(frac, speed, stage):
                if time.monotonic() - last[0] > 1:
                    last[0] = time.monotonic()
                    with lock:
                        print(f"  {name}: {stage} {frac * 100:5.1f}%  {speed or 0:.1f}x", flush=True)
            r = E.encode(info, plan, s, out, trim, cb)
            with lock:
                flag = "  (OVER TARGET)" if r.over_target else ""
                print(f"  {name}: done -> {os.path.basename(r.output)}  {E.fmt_size(r.size)}"
                      f"  {r.elapsed:.1f}s  {r.speed:.1f}x{flag}")
        except Exception as ex:  # noqa: BLE001
            with lock:
                print(f"  {name}: FAILED - {ex}", file=sys.stderr)

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        list(ex.map(run, files))
    print(f"all done in {time.monotonic() - t0:.1f}s")
    if not a.dry_run:  # remember how much each encoder over/undershoots
        cfg["calibration"] = E.calibration()
        try:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        except OSError:
            pass


if __name__ == "__main__":
    main()
