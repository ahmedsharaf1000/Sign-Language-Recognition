import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

print("="*70)
print("ASL Classifier Training")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    print(f"    ✓ Data loaded successfully")
except FileNotFoundError:
    print("    ✗ Error: 'data.pickle' not found!")
    print("    Run 'create_dataset.py' first to generate the dataset")
    exit(1)

data = data_dict['data']
labels = data_dict['labels']

print(f"    Total samples: {len(data)}")
print(f"    Total labels: {len(labels)}")

# Check data consistency
print("\n[2/5] Checking data consistency...")
data_lengths = [len(d) for d in data]
length_counts = Counter(data_lengths)

print(f"    Different data lengths found: {len(length_counts)}")
for length, count in sorted(length_counts.items()):
    print(f"      - Length {length}: {count} samples")

# Filter data to keep only consistent length
most_common_length = length_counts.most_common(1)[0][0]
print(f"\n    Using samples with length: {most_common_length}")

filtered_data = []
filtered_labels = []

for d, l in zip(data, labels):
    if len(d) == most_common_length:
        filtered_data.append(d)
        filtered_labels.append(l)

print(f"    ✓ Filtered dataset: {len(filtered_data)} samples")
print(f"    ✗ Removed: {len(data) - len(filtered_data)} samples")

if len(filtered_data) == 0:
    print("\n    ✗ ERROR: No valid samples found!")
    exit(1)

# Convert to numpy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Show class distribution
print("\n[3/5] Class distribution:")
label_counts = Counter(labels)
for label, count in sorted(label_counts.items()):
    print(f"    {label}: {count} samples")

# Check if we have enough samples per class
min_samples = min(label_counts.values())
if min_samples < 2:
    print(f"\n    ✗ ERROR: Some classes have less than 2 samples!")
    print("    Cannot perform train/test split")
    exit(1)

# Split data
print("\n[4/5] Splitting data (80% train, 20% test)...")
try:
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42
    )
    print(f"    ✓ Training samples: {len(x_train)}")
    print(f"    ✓ Testing samples: {len(x_test)}")
except ValueError as e:
    print(f"    ✗ Error splitting data: {e}")
    print("\n    Trying without stratification...")
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )
    print(f"    ✓ Training samples: {len(x_train)}")
    print(f"    ✓ Testing samples: {len(x_test)}")

# Train model
print("\n[5/5] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)
print("    ✓ Model trained successfully")

# Evaluate model
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Training accuracy
y_train_pred = model.predict(x_train)
train_score = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_score * 100:.2f}%")

# Testing accuracy
y_test_pred = model.predict(x_test)
test_score = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_score * 100:.2f}%")

# Detailed classification report
print("\nDetailed Classification Report:")
print("-"*70)
print(classification_report(y_test, y_test_pred))

# Save model
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✓ Model saved to 'model.p'")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Model Type: Random Forest")
print(f"✓ Features: {data.shape[1]}")
print(f"✓ Classes: {len(set(labels))}")
print(f"✓ Training Samples: {len(x_train)}")
print(f"✓ Testing Samples: {len(x_test)}")
print(f"✓ Test Accuracy: {test_score * 100:.2f}%")
print("="*70)

if test_score < 0.7:
    print("\n⚠ WARNING: Low accuracy! Consider:")
    print("  - Collecting more training data")
    print("  - Ensuring consistent hand positions")
    print("  - Better lighting in images")
    print("  - Clear hand gestures")
elif test_score >= 0.9:
    print("\n✅ Excellent accuracy! Model is ready to use.")
else:
    print("\n✓ Good accuracy! Model should work well.")

print("\nNext step: Run inference_classifier.py to test the model")