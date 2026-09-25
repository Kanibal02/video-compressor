# Video Compressor

Compress videos to a target size (Discord 10 / 50 / 500 MB etc.) using the GPU.
It works out the resolution, framerate and bitrate for you, and you can lock any of them.

## Run

- **GUI:** double-click `Video Compressor.pyw` (or `python "Video Compressor.pyw"`).
  Drag videos or folders onto the window, choose a size, and click **Compress**.
- **CLI:** `python compress.py clip.mp4 -s 10`  (see `python compress.py -h`)

Requires `ffmpeg`/`ffprobe` on PATH and `pip install -r requirements.txt` (PySide6).

## Why it's fast

Measured on an RTX 4070 Super, full clips, with OBS's replay buffer running in the background:

| Clip | Target | Old default (p5) | Now (p4) |
|---|---|---|---|
| 1440p60 HEVC OBS recording, 1:13 | 100 MB | 1.97x | **2.95x** |
| 1080p60 H.264 edit, 0:56 | 25 MB | 2.63x | **3.97x** |

The GPU's encoder chip is the bottleneck. Anything else using it slows compression down, and
OBS's replay buffer alone was using ~40% of it. The app warns you in the status bar when that happens.

- Encodes with **NVENC** (H.264 / HEVC / AV1) instead of CPU 2-pass x264.
- Defaults to NVENC preset p4. On H.264, p5+ was ~40% slower and scored no better on VMAF.
- Decodes on all CPU cores, then **scales on the GPU** (`scale_cuda`) and drops frames before upload.
  Heavy 4K HEVC/AV1 sources automatically use NVDEC instead.
- Runs **2 files in parallel** by default (≈1.5x more batch throughput).
- Content analysis runs in the background as soon as files are added, so it doesn't delay the start.
- If an encode overshoots the target it re-encodes with a corrected bitrate. The app also learns
  each encoder's overshoot and saves it in `settings.json`, so later encodes usually fit on the
  first try.

## How it picks resolution / fps

1. **Smart analysis** encodes 5 x 1 s samples at a fixed quantizer to measure how hard the footage is
   to compress (fast gameplay vs. a static screen recording).
2. It models the bitrate the content needs at any resolution/fps and compares that to your budget.
3. It picks the best combination based on two sliders:
   - **Keep smoothness ↔ Keep detail**: drop resolution to keep 60 fps, or drop fps to keep 1440p.
   - **Cleaner frames ↔ More pixels**: downscale more for artifact-free frames, or keep more
     resolution and accept some blockiness.
4. Locks and limits (Res / FPS tab): lock to source, fixed value, or auto within a min/max range.
   You can also edit the fps list Auto picks from (`60, 50, 48, 30, 25, 24, ...`) and tell it to
   prefer even divisions (60→30 instead of 60→48) to avoid judder.

Hover a row's **Output plan** to see the full plan: exact size, bitrates, content complexity, predicted quality.

## Handy stuff

- Right-click a finished file → **Copy output file**, then Ctrl+V straight into Discord.
- Right-click → **Set trim...** to compress only part of a clip (or double-click the Trim cell).
- Audio: first track / **mix all tracks** (OBS game + mic) / keep all / none. AAC or Opus.
- Target modes: file size, % of original, bitrate, or constant quality.
- Converts HDR phone clips to SDR, handles rotated phone videos, supports MP4/MKV output.
