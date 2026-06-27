"""
main.py — ADAS Simulator entry point

Usage:
    python main.py                                    # webcam, all features
    python main.py --source videos/dashcam2.mp4
    python main.py --source videos/dashcam2.mp4 --save
    python main.py --source videos/dashcam2.mp4 --no-depth   # skip depth (faster)
"""

import argparse
from services.video_processor import VideoProcessor


def parse_args():
    p = argparse.ArgumentParser(description="ADAS Simulator")
    p.add_argument("--source",    type=str, default="0")
    p.add_argument("--save",      action="store_true")
    p.add_argument("--no-depth",  action="store_true",
                   help="Disable depth estimation (faster, no transformers needed)")
    return p.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source == "0" else args.source
    VideoProcessor(
        source=source,
        save_output=args.save,
        enable_depth=not args.no_depth,
    ).run()


if __name__ == "__main__":
    main()
