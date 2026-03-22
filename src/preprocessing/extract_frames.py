"""
extract_frames.py
-----------------
Splits MBG videos into image frames for training.

Usage:
    python3 src/preprocessing/extract_frames.py \
        --video_dir data/raw/videos \
        --output_dir data/frames \
        --fps 1
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> int:
    """
    Extracts frames from a video at the specified frame rate.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        fps: Number of frames to extract per second.

    Returns:
        Total number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))
    video_name = Path(video_path).stem

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_file = os.path.join(output_dir, f"{video_name}_frame_{saved:06d}.jpg")
            cv2.imwrite(frame_file, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"[INFO] {video_path} -> {saved} frames saved to {output_dir}")
    return saved


def process_all_videos(video_dir: str, output_dir: str, fps: int = 1):
    """
    Processes all .mp4 and .avi videos in a directory.
    """
    video_dir = Path(video_dir)
    total = 0
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        for video_path in sorted(video_dir.glob(ext)):
            sub_dir = Path(output_dir) / video_path.stem
            total += extract_frames(str(video_path), str(sub_dir), fps)
    print(f"\n[INFO] Total frames extracted: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MBG videos.")
    parser.add_argument("--video_dir", type=str, default="data/raw/videos",
                        help="Directory containing videos.")
    parser.add_argument("--output_dir", type=str, default="data/frames",
                        help="Output directory for extracted frames.")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second to extract.")
    args = parser.parse_args()
    process_all_videos(args.video_dir, args.output_dir, args.fps)
