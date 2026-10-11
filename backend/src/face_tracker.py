"""
face_tracker.py — MediaPipe-powered face + gesture tracker for SupoClip.

Uses MediaPipe's FaceDetection (fast, model_selection=1 for full-range) and
Holistic (face mesh + hands + pose) to compute the Region of Interest (ROI)
of the presenter per-frame. Applies Exponential Moving Average (EMA) smoothing
so the dynamic pan follows fast movements without jitter.

Detection priority:
    1. MediaPipe FaceDetection  (fast, low overhead)
       + MediaPipe Hands        (gesture/arm ROI expansion)
    2. OpenCV Haar cascade      (fallback if MediaPipe unavailable)

Usage:
    tracker = FaceTracker.get_instance()
    roi_sequence = tracker.track_video(video_path, start_sec, end_sec)
    # roi_sequence: List[(t, cx_norm, cy_norm, w_norm, h_norm)]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# EMA smoothing factor — higher = more responsive to fast movement,
# lower = smoother/less jitter. 0.6 is a good balance for gestures.
_DEFAULT_EMA_ALPHA: float = float(os.environ.get("MEDIAPIPE_EMA_ALPHA", "0.6"))
_DEFAULT_CONFIDENCE: float = float(
    os.environ.get("MEDIAPIPE_CONFIDENCE", "0.5")
)

# How many frames to sample per second of clip
_FRAMES_PER_SECOND: int = int(os.environ.get("MEDIAPIPE_FPS_SAMPLE", "4"))

# ROI padding ratio around the detected body region (adds headroom above head,
# arm space on sides, etc.)
_ROI_PADDING: float = 0.12


@dataclass
class ROIFrame:
    """Detected region-of-interest for a single video frame."""

    time: float          # seconds from start of video
    cx: float            # normalised center-x  (0.0 – 1.0)
    cy: float            # normalised center-y  (0.0 – 1.0)
    w: float             # normalised width     (0.0 – 1.0)
    h: float             # normalised height    (0.0 – 1.0)
    confidence: float


class FaceTracker:
    """
    Singleton face + gesture tracker.  Initialised once per worker process so
    MediaPipe models are loaded only once.
    """

    _instance: Optional["FaceTracker"] = None

    def __init__(self, confidence: float = _DEFAULT_CONFIDENCE) -> None:
        self._confidence = confidence
        self._mp_face = None
        self._mp_hands = None
        self._haar = None
        self._mediapipe_ok = False
        self._init_detectors()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "FaceTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Force re-initialisation (useful in tests)."""
        if cls._instance is not None:
            cls._instance._close()
            cls._instance = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_detectors(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            # FaceDetection — model 1 = full-range (good up to ~5 m)
            self._mp_face = mp.solutions.face_detection.FaceDetection(  # type: ignore
                model_selection=1,
                min_detection_confidence=self._confidence,
            )

            # Hands — tracks up to 2 hands; captures gesture ROI
            self._mp_hands = mp.solutions.hands.Hands(  # type: ignore
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self._confidence,
                min_tracking_confidence=0.4,
            )

            self._mediapipe_ok = True
            logger.info(
                "FaceTracker: MediaPipe FaceDetection + Hands initialised "
                f"(confidence={self._confidence})"
            )
        except ImportError:
            logger.warning(
                "FaceTracker: mediapipe not importable — using OpenCV Haar fallback"
            )
        except Exception as exc:
            logger.warning(f"FaceTracker: MediaPipe init failed ({exc}) — using Haar fallback")

        # Always load Haar cascade as ultimate fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            logger.info("FaceTracker: Haar cascade loaded as fallback")
        except Exception as exc:
            logger.error(f"FaceTracker: Haar cascade failed to load: {exc}")

    def _close(self) -> None:
        try:
            if self._mp_face:
                self._mp_face.close()
            if self._mp_hands:
                self._mp_hands.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_video(
        self,
        video_path: Path | str,
        start_sec: float,
        end_sec: float,
        ema_alpha: float = _DEFAULT_EMA_ALPHA,
        fps_sample: int = _FRAMES_PER_SECOND,
    ) -> List[ROIFrame]:
        """
        Sample frames from the clip [start_sec, end_sec], detect face + hands
        per frame, expand to a unified ROI, apply EMA smoothing, and return
        a list of ROIFrame ordered by time.

        Args:
            video_path:   path to the source video file.
            start_sec:    clip start (seconds).
            end_sec:      clip end (seconds).
            ema_alpha:    EMA smoothing factor (0=fully smooth, 1=no smoothing).
            fps_sample:   frames sampled per second of clip.

        Returns:
            List[ROIFrame] — one entry per sampled frame, smoothed.
            Empty list if detection fails entirely.
        """
        frames_data = self._extract_frames(video_path, start_sec, end_sec, fps_sample)
        if not frames_data:
            logger.warning("FaceTracker: no frames extracted — returning empty ROI")
            return []

        raw_rois: List[Optional[ROIFrame]] = []
        for t, frame_rgb in frames_data:
            roi = self._detect_roi_in_frame(frame_rgb, t)
            raw_rois.append(roi)

        smoothed = self._apply_ema(raw_rois, ema_alpha)
        logger.info(
            f"FaceTracker: tracked {len(smoothed)} ROI frames "
            f"({sum(1 for r in smoothed if r is not None)} detected, "
            f"{sum(1 for r in smoothed if r is None)} interpolated)"
        )
        return [r for r in smoothed if r is not None]

    def get_static_crop_center(
        self,
        video_path: Path | str,
        start_sec: float,
        end_sec: float,
    ) -> Tuple[float, float]:
        """
        Returns the weighted-average (cx, cy) in normalised coords across the
        clip — useful as a fallback when dynamic pan is not desired.
        """
        rois = self.track_video(video_path, start_sec, end_sec, ema_alpha=0.3)
        if not rois:
            return 0.5, 0.4  # default slightly above center

        total_w = sum(r.confidence for r in rois)
        cx = sum(r.cx * r.confidence for r in rois) / max(total_w, 1e-6)
        cy = sum(r.cy * r.confidence for r in rois) / max(total_w, 1e-6)
        return cx, cy

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------
    def _extract_frames(
        self,
        video_path: Path | str,
        start_sec: float,
        end_sec: float,
        fps_sample: int,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames directly via OpenCV (faster than MoviePy for random seeks).
        Returns list of (timestamp_sec, frame_rgb).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"FaceTracker: cannot open video {video_path}")
            return []

        try:
            duration = end_sec - start_sec
            n_frames = max(1, int(duration * fps_sample))
            # Always include start, middle and end; fill remaining evenly
            sample_times = sorted(
                set(
                    [start_sec, start_sec + duration / 2, min(end_sec - 0.05, end_sec)]
                    + list(np.linspace(start_sec, end_sec - 0.05, n_frames))
                )
            )

            result: List[Tuple[float, np.ndarray]] = []
            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result.append((t, frame_rgb))

            logger.debug(f"FaceTracker: extracted {len(result)} frames from [{start_sec:.1f}–{end_sec:.1f}]s")
            return result
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Per-frame detection
    # ------------------------------------------------------------------
    def _detect_roi_in_frame(
        self, frame_rgb: np.ndarray, t: float
    ) -> Optional[ROIFrame]:
        """
        Detect face + hands in a single RGB frame.
        Returns a ROIFrame with the unified bounding region, or None.
        """
        h, w = frame_rgb.shape[:2]

        boxes: List[Tuple[float, float, float, float, float]] = []
        # (x_min_norm, y_min_norm, x_max_norm, y_max_norm, confidence)

        if self._mediapipe_ok:
            boxes.extend(self._run_mediapipe_face(frame_rgb, w, h))
            boxes.extend(self._run_mediapipe_hands(frame_rgb, w, h))

        if not boxes:
            boxes.extend(self._run_haar(frame_rgb, w, h))

        if not boxes:
            return None

        # Merge all boxes into one bounding region
        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[2] for b in boxes)
        y_max = max(b[3] for b in boxes)
        avg_conf = sum(b[4] for b in boxes) / len(boxes)

        # Add padding
        pad_x = _ROI_PADDING * (x_max - x_min)
        pad_y = _ROI_PADDING * (y_max - y_min)
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(1.0, x_max + pad_x)
        y_max = min(1.0, y_max + pad_y)

        roi_w = x_max - x_min
        roi_h = y_max - y_min
        cx = x_min + roi_w / 2.0
        cy = y_min + roi_h / 2.0

        # Bias the center slightly upward so there's headroom above the face
        cy = max(0.1, cy - 0.05)

        return ROIFrame(time=t, cx=cx, cy=cy, w=roi_w, h=roi_h, confidence=avg_conf)

    def _run_mediapipe_face(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """Run MediaPipe FaceDetection and return normalised boxes."""
        boxes = []
        try:
            results = self._mp_face.process(frame_rgb)  # type: ignore
            if results and results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    conf = det.score[0] if det.score else self._confidence
                    x_min = max(0.0, bbox.xmin)
                    y_min = max(0.0, bbox.ymin)
                    x_max = min(1.0, bbox.xmin + bbox.width)
                    y_max = min(1.0, bbox.ymin + bbox.height)
                    if bbox.width > 0.01 and bbox.height > 0.01:
                        boxes.append((x_min, y_min, x_max, y_max, float(conf)))
        except Exception as exc:
            logger.debug(f"FaceTracker: MediaPipe face error: {exc}")
        return boxes

    def _run_mediapipe_hands(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """Run MediaPipe Hands and return normalised bounding boxes per hand."""
        boxes = []
        try:
            results = self._mp_hands.process(frame_rgb)  # type: ignore
            if results and results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    xs = [lm.x for lm in hand_lms.landmark]
                    ys = [lm.y for lm in hand_lms.landmark]
                    x_min = max(0.0, min(xs))
                    y_min = max(0.0, min(ys))
                    x_max = min(1.0, max(xs))
                    y_max = min(1.0, max(ys))
                    if (x_max - x_min) > 0.01 and (y_max - y_min) > 0.01:
                        # Lower confidence weight for hands vs. face
                        boxes.append((x_min, y_min, x_max, y_max, 0.6))
        except Exception as exc:
            logger.debug(f"FaceTracker: MediaPipe hands error: {exc}")
        return boxes

    def _run_haar(
        self, frame_rgb: np.ndarray, w: int, h: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """Haar cascade fallback — returns normalised boxes."""
        boxes = []
        if self._haar is None:
            return boxes
        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(max(20, w // 20), max(20, h // 20)),
            )
            for (fx, fy, fw, fh) in faces:
                boxes.append((
                    fx / w,
                    fy / h,
                    (fx + fw) / w,
                    (fy + fh) / h,
                    0.5,
                ))
        except Exception as exc:
            logger.debug(f"FaceTracker: Haar error: {exc}")
        return boxes

    # ------------------------------------------------------------------
    # EMA smoothing
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_ema(
        rois: List[Optional[ROIFrame]],
        alpha: float,
    ) -> List[Optional[ROIFrame]]:
        """
        Apply Exponential Moving Average across cx/cy/w/h.
        Fills None gaps with the nearest-neighbour before smoothing so
        fast gestures/movement don't create dead zones.
        """
        if not rois:
            return rois

        # Forward-fill Nones from last known value
        filled: List[Optional[ROIFrame]] = []
        last = None
        for r in rois:
            if r is not None:
                last = r
            filled.append(last if r is None else r)

        # Backward-fill for leading Nones
        last = None
        for i in range(len(filled) - 1, -1, -1):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last

        if all(r is None for r in filled):
            return rois

        smoothed: List[Optional[ROIFrame]] = []
        prev = None
        for r in filled:
            if r is None:
                smoothed.append(None)
                continue
            if prev is None:
                smoothed.append(r)
                prev = r
                continue
            # EMA blend
            new_cx = alpha * r.cx + (1 - alpha) * prev.cx
            new_cy = alpha * r.cy + (1 - alpha) * prev.cy
            new_w  = alpha * r.w  + (1 - alpha) * prev.w
            new_h  = alpha * r.h  + (1 - alpha) * prev.h
            blended = ROIFrame(
                time=r.time,
                cx=new_cx, cy=new_cy,
                w=new_w, h=new_h,
                confidence=r.confidence,
            )
            smoothed.append(blended)
            prev = blended

        return smoothed
