"""
perception/face/face_worker.py

Responsibility:
    Transform an image into a FaceObservation.

This worker coordinates:
    - Face detection
    - Emotion analysis
    - Observation construction

It DOES NOT:
    - Draw on images
    - Display UI
    - Update the world model
    - Publish events
    - Perform identity recognition
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from perception.camera_frame.detector import FaceDetector
from perception.camera_frame.emotion_analyzer import EmotionAnalyzer
from perception.camera_frame.models import Face, FaceObservation


class FaceWorker:
    """
    Processes an image and extracts face-related observations.
    """

    def __init__(
        self,
        detector: FaceDetector,
        emotion_analyzer: EmotionAnalyzer,
    ) -> None:
        self._detector = detector
        self._emotion_analyzer = emotion_analyzer

    async def process(
        self,
        image: np.ndarray,
    ) -> FaceObservation:
        """
        Analyze an image for faces and emotions.

        Args:
            image:
                BGR OpenCV image.

        Returns:
            FaceObservation
        """

        detections = self._detector.detect(image)

        faces: list[Face] = []

        for detection in detections:
            x, y, w, h = detection.bbox

            face_crop = image[y:y + h, x:x + w]

            emotion = self._emotion_analyzer.analyze(face_crop)

            faces.append(
                Face(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    emotion=emotion.dominant_emotion,
                    emotion_scores=emotion.scores,
                )
            )

        return FaceObservation(
            timestamp=datetime.now(timezone.utc),
            face_count=len(faces),
            multiple_faces=len(faces) > 1,
            faces=faces,
        )


if __name__ == "__main__":
    from perception.face.detector import MediaPipeFaceDetector
    from perception.face.emotion_analyzer import DeepFaceEmotionAnalyzer

    detector = MediaPipeFaceDetector()
    emotion = DeepFaceEmotionAnalyzer()

    worker = FaceWorker(
        detector=detector,
        emotion_analyzer=emotion,
    )

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        observation = worker.process(frame)

        print(observation)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()