"""
main.py
Entry point for the ADAS Simulator.

Usage examples:
    python main.py                          # webcam
    python main.py --source videos/clip.mp4
    python main.py --source videos/clip.mp4 --save
"""

import argparse
import sys
from services.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADAS Simulator — Milestone 1: Detection & Tracking"
    )
    p.add_argument(
        "--source", type=str, default="0",
        help="Video file path or '0' for webcam (default: 0)"
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save annotated video to outputs/"
    )
    return p.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source == "0" else args.source
    processor = VideoProcessor(source=source, save_output=args.save)
    processor.run()


if __name__ == "__main__":
    main()
