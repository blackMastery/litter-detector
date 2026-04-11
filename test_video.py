# ─────────────────────────────────────────────
#  test_video.py  —  Phase 0 Video File Tester
#
#  Run against a local video file to validate
#  the full detection pipeline before connecting
#  a live camera.
#
#  Usage:
#    python test_video.py                         # uses TEST_VIDEO_PATH in config.py
#    python test_video.py --video myclip.mp4      # specify a video file
#    python test_video.py --video myclip.mp4 --show          # show live window
#    python test_video.py --video myclip.mp4 --show --slow   # slow-motion playback
#    python test_video.py --download-sample       # download a free sample street video
# ─────────────────────────────────────────────

import argparse
import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Allow running from any directory ──────────
sys.path.insert(0, str(Path(__file__).parent))

from detector import LitterDetector, DumpEvent
from config import (
    CONFIDENCE_THRESHOLD, SNAPSHOT_DIR, LOG_DIR,
    LITTER_CLASSES, PERSON_CLASS, PROXIMITY_THRESHOLD,
)
import logger as event_logger


# ─────────────────────────────────────────────
#  Terminal colours (no external deps)
# ─────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    AMBER  = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

def g(s): return f"{C.GREEN}{s}{C.RESET}"
def a(s): return f"{C.AMBER}{s}{C.RESET}"
def r(s): return f"{C.RED}{s}{C.RESET}"
def b(s): return f"{C.BLUE}{s}{C.RESET}"
def dim(s): return f"{C.GRAY}{s}{C.RESET}"
def bold(s): return f"{C.BOLD}{s}{C.RESET}"


# ─────────────────────────────────────────────
#  Progress bar
# ─────────────────────────────────────────────

def progress_bar(current, total, width=40):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.0%}  frame {current}/{total}"


# ─────────────────────────────────────────────
#  Sample video downloader
# ─────────────────────────────────────────────

def download_sample_video():
    """
    Downloads a free Creative Commons street footage clip.
    Uses a publicly available short MP4 from Pexels/pixabay CDN.
    """
    import urllib.request

    # Short (~10s) street scene — CC0 licence
    url = (
        "https://download.samplelib.com/mp4/sample-5s.mp4"
    )
    dest = "sample_street.mp4"

    if Path(dest).exists():
        print(g(f"✔ Sample already exists: {dest}"))
        return dest

    print(f"Downloading sample video to {bold(dest)} …")
    try:
        urllib.request.urlretrieve(url, dest,
            reporthook=lambda b, bs, ts: print(
                f"\r  {progress_bar(b*bs, ts)}", end="", flush=True))
        print()
        print(g(f"✔ Downloaded: {dest}"))
        return dest
    except Exception as e:
        print(r(f"✘ Download failed: {e}"))
        print(a("  Try providing your own video with --video path/to/file.mp4"))
        sys.exit(1)


# ─────────────────────────────────────────────
#  Session summary report
# ─────────────────────────────────────────────

def print_summary(results: dict):
    divider = dim("─" * 60)
    print(f"\n{divider}")
    print(bold("  TEST SUMMARY"))
    print(divider)

    print(f"  Video file      : {bold(results['video_path'])}")
    print(f"  Duration        : {results['duration']:.1f}s")
    print(f"  Frames tested   : {results['frames_tested']}")
    print(f"  Frames skipped  : {results['frames_skipped']}  (every {results['skip']}th frame)")
    print(f"  Avg inference   : {results['avg_ms']:.1f} ms/frame  "
          f"({1000/max(results['avg_ms'],1):.1f} FPS)")
    print(divider)
    print(f"  {g('✔')} Litter detections : {bold(str(results['litter_count']))}")
    print(f"  {b('✔')} Person detections  : {bold(str(results['person_count']))}")
    if results['dump_count']:
        print(f"  {r('⚠')} Dumping events     : {bold(str(results['dump_count']))}  ← snapshots saved")
    else:
        print(f"  {dim('○')} Dumping events     : 0")
    print(divider)

    if results['dump_count']:
        print(f"\n  {r('Dumping incidents:')}")
        for ev in results['dump_events']:
            print(f"    {dim('•')} {ev['timestamp']}  |  {ev['litter_label']}  "
                  f"|  snap: {dim(ev['snapshot_path'])}")

    if results['snapshots']:
        print(f"\n  {bold('Snapshots saved to:')} {SNAPSHOT_DIR}/")
        for s in results['snapshots']:
            print(f"    {dim('•')} {Path(s).name}")

    print(f"\n  {bold('CSV logs written to:')} {LOG_DIR}/")
    print(f"    {dim('•')} detections.csv")
    print(f"    {dim('•')} incidents.csv")
    print(f"\n{divider}\n")


# ─────────────────────────────────────────────
#  Core test runner
# ─────────────────────────────────────────────

def run_test(video_path: str, show: bool, slow: bool,
             confidence: float, skip: int, max_frames: int):

    print(f"\n{bold('Phase 0 — Litter & Dumping Detector  |  Video Test')}")
    print(dim("─" * 60))

    # ── Open video ────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(r(f"✘ Cannot open video: {video_path}"))
        print(a("  Check the file path and that it is a valid video format."))
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps

    print(f"  {g('✔')} Opened: {bold(video_path)}")
    print(f"  Resolution  : {width}×{height}")
    print(f"  Duration    : {duration:.1f}s  ({total_frames} frames @ {fps:.1f} FPS)")
    print(f"  Confidence  : {confidence:.0%}")
    print(f"  Frame skip  : every {skip}th frame")
    if max_frames:
        print(f"  Max frames  : {max_frames}")
    print()

    # ── Load model ────────────────────────────
    print(f"  Loading YOLOv8 model…", end=" ", flush=True)
    detector = LitterDetector()
    print(g("done"))
    print()

    # ── Output video writer ───────────────────
    os.makedirs("test_output", exist_ok=True)
    out_path = f"test_output/annotated_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / skip, (width, height))

    # ── Setup ─────────────────────────────────
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    litter_count = 0
    person_count = 0
    dump_count   = 0
    dump_events  = []
    snapshots    = []
    inference_times = []
    frame_idx    = 0
    tested       = 0

    if show:
        cv2.namedWindow("Litter Detector — Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Litter Detector — Test", min(width, 1280), min(height, 720))

    print(f"  {'Frame':<8} {'Status':<30} {'Litter':>7} {'Persons':>8} {'Incidents':>10}  ms")
    print(dim("  " + "─" * 72))

    # ── Frame loop ────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Skip frames for speed
        if frame_idx % skip != 0:
            continue

        if max_frames and tested >= max_frames:
            break

        tested += 1
        t0 = time.perf_counter()
        annotated, litter_dets, person_dets, dump_event = detector.process_frame(
            frame.copy(), confidence_override=confidence
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(elapsed_ms)

        # Counts
        litter_count += len(litter_dets)
        person_count += len(person_dets)

        # Log detections
        ts = time.strftime("%H:%M:%S")
        for l in litter_dets:
            event_logger.log_detection(l.label, l.confidence, "litter")
        for p in person_dets:
            event_logger.log_detection(p.label, p.confidence, "person")

        # Dump event
        if dump_event:
            dump_count += 1
            ev_dict = {
                "timestamp":     dump_event.timestamp,
                "litter_label":  dump_event.litter_label,
                "person_conf":   dump_event.person_conf,
                "litter_conf":   dump_event.litter_conf,
                "snapshot_path": dump_event.snapshot_path,
            }
            dump_events.append(ev_dict)
            snapshots.append(dump_event.snapshot_path)
            event_logger.log_incident(
                dump_event.timestamp, dump_event.litter_label,
                dump_event.snapshot_path, dump_event.person_conf, dump_event.litter_conf,
            )

        # Write annotated frame to output video
        writer.write(annotated)

        # ── Print row every 10 tested frames or on event
        if tested % 10 == 0 or dump_event or (litter_dets and tested <= 5):
            status = ""
            if dump_event:
                status = r(f"⚠ DUMP — {dump_event.litter_label}")
            elif litter_dets:
                labels = ", ".join(set(d.label for d in litter_dets))
                status = a(f"litter: {labels}")
            elif person_dets:
                status = b(f"person(s) in frame")
            else:
                status = dim("clear")

            print(
                f"  {str(frame_idx):<8} "
                f"{status:<38} "
                f"{g(str(litter_count)):>7} "
                f"{b(str(person_count)):>8} "
                f"{(r(str(dump_count)) if dump_count else dim('0')):>10}  "
                f"{dim(f'{elapsed_ms:.0f}')}"
            )

        # ── Show window
        if show:
            # Overlay progress
            prog_pct = frame_idx / max(total_frames, 1)
            bar_w = int(width * prog_pct)
            cv2.rectangle(annotated, (0, height-6), (bar_w, height), (30, 180, 100), -1)
            cv2.imshow("Litter Detector — Test", annotated)
            delay = int(1000 / fps) * (3 if slow else 1)
            key = cv2.waitKey(delay)
            if key == ord('q') or key == 27:   # q or ESC to quit early
                print(a("\n  Stopped early by user (q / ESC)."))
                break

    # ── Cleanup ───────────────────────────────
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    avg_ms = np.mean(inference_times) if inference_times else 0

    results = {
        "video_path":     video_path,
        "duration":       duration,
        "frames_tested":  tested,
        "frames_skipped": frame_idx - tested,
        "skip":           skip,
        "avg_ms":         avg_ms,
        "litter_count":   litter_count,
        "person_count":   person_count,
        "dump_count":     dump_count,
        "dump_events":    dump_events,
        "snapshots":      snapshots,
        "output_video":   out_path,
    }

    print_summary(results)
    print(f"  {g('✔')} Annotated video saved to: {bold(out_path)}")
    print(f"      Open it to review all detections with bounding boxes.\n")

    return results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test litter/dumping detector against a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video.py --download-sample
  python test_video.py --video sample_street.mp4
  python test_video.py --video myclip.mp4 --show
  python test_video.py --video myclip.mp4 --show --slow
  python test_video.py --video myclip.mp4 --confidence 0.35 --skip 2
  python test_video.py --video myclip.mp4 --max-frames 100
        """
    )
    parser.add_argument("--video",          type=str,   help="Path to video file")
    parser.add_argument("--show",           action="store_true",
                        help="Show live annotated window while processing")
    parser.add_argument("--slow",           action="store_true",
                        help="Slow-motion playback (3× slower) in --show mode")
    parser.add_argument("--confidence",     type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Detection confidence 0.0–1.0 (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--skip",           type=int,   default=3,
                        help="Process every Nth frame (default: 3, use 1 for every frame)")
    parser.add_argument("--max-frames",     type=int,   default=0,
                        help="Stop after N frames (0 = process entire video)")
    parser.add_argument("--download-sample", action="store_true",
                        help="Download a free sample street video and run test on it")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.download_sample:
        video_path = download_sample_video()
    elif args.video:
        video_path = args.video
    else:
        # Fall back to config.py TEST_VIDEO_PATH
        from config import TEST_VIDEO_PATH
        if TEST_VIDEO_PATH and Path(TEST_VIDEO_PATH).exists():
            video_path = TEST_VIDEO_PATH
        else:
            print(r("✘ No video specified."))
            print(a("  Run with --video path/to/file.mp4"))
            print(a("  Or run with --download-sample to get a free test clip"))
            sys.exit(1)

    if not Path(video_path).exists():
        print(r(f"✘ File not found: {video_path}"))
        sys.exit(1)

    run_test(
        video_path  = video_path,
        show        = args.show,
        slow        = args.slow,
        confidence  = args.confidence,
        skip        = args.skip,
        max_frames  = args.max_frames,
    )
