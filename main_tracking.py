import os
import cv2
import numpy as np
from ultralytics import YOLO

from tracking.memory_botsort import MemoryEnhancedBoTSORT
from reid.feature_extractor import EnhancedFeatureExtractor
from rag.metadata_generator import MetadataGenerator


def main():
    print("=" * 60)
    print("🚀 Starting Intelligent Video Tracking System")
    print("=" * 60)

    # -------------------------------
    # CONFIG
    # -------------------------------
    VIDEO_PATH = "input.mp4"
    OUTPUT_METADATA = "data/metadata.jsonl"
    CONF_THRESHOLD = 0.4   # 🔥 updated threshold
    MAX_FRAMES = None      # set to limit frames if needed

    # -------------------------------
    # CHECK VIDEO
    # -------------------------------
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error opening video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Total Frames: {total_frames}")

    # -------------------------------
    # INIT MODELS
    # -------------------------------
    print("\n🔍 Loading YOLO model...")
    detector = YOLO("yolo11n.pt")
    print("✅ YOLO loaded")

    print("\n🧠 Loading Feature Extractor...")
    feature_extractor = EnhancedFeatureExtractor()
    print("✅ Feature extractor ready")

    print("\n📊 Initializing Tracker...")
    tracker = MemoryEnhancedBoTSORT(feature_extractor=feature_extractor)
    print("✅ Tracker initialized")

    print("\n📝 Initializing Metadata Generator...")
    metadata = MetadataGenerator()
    print("✅ Metadata generator ready")

    print("\n" + "=" * 60)
    print("▶️ Processing Video...")
    print("=" * 60)

    frame_idx = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if MAX_FRAMES and frame_idx >= MAX_FRAMES:
            break

        frame_idx += 1

        # -------------------------------
        # DETECTION
        # -------------------------------
        results = detector(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                if conf > CONF_THRESHOLD:
                    detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if detections else np.empty((0, 5))
        total_detections += len(detections)

        # -------------------------------
        # TRACKING
        # -------------------------------
        tracks = tracker.update(frame, detections)

        # -------------------------------
        # METADATA STORAGE
        # -------------------------------
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            metadata.add(
                frame=frame_idx,
                track_id=int(track_id),
                bbox=[x1, y1, x2, y2]
            )

        # -------------------------------
        # DEBUG PRINT
        # -------------------------------
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: {len(tracks)} active tracks | {len(detections)} detections")

    # -------------------------------
    # SAVE METADATA
    # -------------------------------
    os.makedirs("data", exist_ok=True)
    metadata.save(OUTPUT_METADATA)

    cap.release()

    print("\n" + "=" * 60)
    print("✅ TRACKING COMPLETE")
    print("=" * 60)
    print(f"📊 Total Frames Processed: {frame_idx}")
    print(f"📦 Total Detections: {total_detections}")
    print(f"💾 Metadata saved to: {OUTPUT_METADATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()