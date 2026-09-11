"""Streaming frame extraction from a video file."""

import os

import cv2
import numpy as np


class VideoFrameExtractor:
    """Iterate a video's frames one at a time, with 1-based frame IDs.

    Yields `(frame_id, bgr, meta)`:
        frame_id: 1-based ID (from `start_index`) -- keys the output trajectory.
        bgr:      uint8 HxWx3, native resolution.
        meta:     {'frame_id', 'position', 'width', 'height'}.
    """

    def __init__(self, video_path, start_index=1):
        video_path = os.path.abspath(os.path.expanduser(str(video_path)))
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.start_index = int(start_index)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        try:
            # Container metadata only, no decode.
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

    def __len__(self):
        """Advisory length, from the container's frame count."""
        return self.frame_count

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        try:
            position = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_id = self.start_index + position
                meta = {
                    'frame_id': frame_id,
                    'position': position,
                    'width': int(frame.shape[1]),
                    'height': int(frame.shape[0]),
                }
                position += 1
                # Fresh buffer per read() -- safe to yield without copying.
                yield frame_id, frame, meta
        finally:
            cap.release()

def probe_video(video_path):
    """Container metadata without decoding any frames."""
    extractor = VideoFrameExtractor(video_path)
    return {
        'path': extractor.video_path,
        'frame_count': extractor.frame_count,
        'fps': extractor.fps,
        'width': extractor.width,
        'height': extractor.height,
    }
