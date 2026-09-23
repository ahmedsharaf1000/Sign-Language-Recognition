#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASL Dataset Creator - Fixed Version
Works with MediaPipe 0.10.x
Only captures ONE hand per image for consistency
"""

import os
import sys
import pickle

print("=" * 70)
print("ASL Dataset Creator")
print("=" * 70)

# Step 1: Import and verify MediaPipe
print("\n[1/6] Importing MediaPipe...")
try:
    import mediapipe as mp

    print(f"    ✓ MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"    ✗ Error: {e}")
    print("\n    Please install: pip install mediapipe==0.10.9")
    sys.exit(1)

# Step 2: Verify solutions module
print("\n[2/6] Verifying mp.solutions...")
if not hasattr(mp, 'solutions'):
    print("    ✗ ERROR: mp.solutions not found!")
    print("\n    Possible causes:")
    print("    1. Wrong MediaPipe version")
    print("    2. File conflict (mediapipe.py in current folder)")
    print("\n    Solutions:")
    print("    1. Delete any 'mediapipe.py' file here")
    print("    2. Reinstall: pip uninstall mediapipe && pip install mediapipe==0.10.9")
    sys.exit(1)
print("    ✓ mp.solutions found")

# Step 3: Import OpenCV
print("\n[3/6] Importing OpenCV...")
try:
    import cv2

    print(f"    ✓ OpenCV version: {cv2.__version__}")
except ImportError:
    print("    ✗ OpenCV not found")
    print("    Install: pip install opencv-python")
    sys.exit(1)

# Step 4: Initialize MediaPipe Hands
print("\n[4/6] Initializing MediaPipe Hands...")
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # IMPORTANT: max_num_hands=1 ensures we only detect one hand
    hands = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.3,
        max_num_hands=1  # Only detect ONE hand for consistency
    )
    print("    ✓ Hands detector initialized (max 1 hand per image)")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)

# Step 5: Check data directory
print("\n[5/6] Checking data directory...")
DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    print(f"    ✗ Directory '{DATA_DIR}' not found!")
    print("\n    Create folder structure:")
    print("    data/")
    print("    ├── A/")
    print("    │   ├── img1.jpg")
    print("    │   └── img2.jpg")
    print("    ├── B/")
    print("    └── C/")
    sys.exit(1)

# Count classes
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if not classes:
    print(f"    ✗ No class folders found in '{DATA_DIR}'")
    sys.exit(1)

print(f"    ✓ Found {len(classes)} classes: {', '.join(sorted(classes))}")

# Step 6: Process images
print("\n[6/6] Processing images...")
print("-" * 70)

data = []
labels = []
total_processed = 0
total_failed = 0

for class_idx, dir_ in enumerate(sorted(classes), 1):
    dir_path = os.path.join(DATA_DIR, dir_)

    images = [f for f in os.listdir(dir_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"\n[{class_idx}/{len(classes)}] Class '{dir_}' ({len(images)} images)")

    class_success = 0
    class_failed = 0

    for img_name in images:
        img_path = os.path.join(dir_path, img_name)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"    ⚠ Could not read: {img_name}")
            class_failed += 1
            continue

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Get ONLY the first hand (guaranteed to be 1 hand due to max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract coordinates
            x_coords = []
            y_coords = []

            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)

            # Normalize coordinates
            min_x = min(x_coords)
            min_y = min(y_coords)

            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Verify we have exactly 42 features (21 landmarks * 2)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
                class_success += 1
                total_processed += 1
            else:
                print(f"    ⚠ Unexpected feature count: {len(data_aux)} for {img_name}")
                class_failed += 1
        else:
            class_failed += 1

    print(f"    ✓ Success: {class_success}, ✗ Failed: {class_failed}")
    total_failed += class_failed

# Cleanup
hands.close()

# Verify data consistency
print("\n" + "=" * 70)
print("DATA VERIFICATION")
print("=" * 70)

data_lengths = set([len(d) for d in data])
print(f"Different feature lengths: {data_lengths}")

if len(data_lengths) == 1:
    print(f"✓ All samples have consistent length: {list(data_lengths)[0]} features")
else:
    print(f"⚠ WARNING: Inconsistent data lengths found!")
    print("This should not happen with the fixed code.")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total images processed: {total_processed}")
print(f"Total images failed: {total_failed}")
print(f"Total classes: {len(set(labels))}")
print(f"Dataset size: {len(data)} samples")
print(f"Features per sample: {len(data[0]) if data else 0}")

# Save dataset
if len(data) > 0:
    print("\nSaving dataset...")
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("✓ Dataset saved to 'data.pickle'")

    # Show class distribution
    print("\nClass distribution:")
    from collections import Counter

    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")

    print("\n" + "=" * 70)
    print("✅ SUCCESS! Dataset created successfully!")
    print("=" * 70)
    print("\nNext step: Run train_classifier.py to train the model")
else:
    print("\n" + "=" * 70)
    print("⚠ WARNING: No data collected!")
    print("=" * 70)
    print("\nPossible reasons:")
    print("1. No hands detected in images")
    print("2. Images are too dark/blurry")
    print("3. Hands are too small in frame")
    print("\nTips:")
    print("- Ensure hands are clearly visible")
    print("- Use good lighting")
    print("- Hands should fill ~50% of frame")
    sys.exit(1)