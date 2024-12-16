from itertools import product  # Added missing import
# from sklearn.model_selection import KFold
from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import os
import random
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_and_split_data(base_path, non_face_dir, modified_face_dir):
    """
    Load all data including:
    - Original AT&T dataset (400 images: 40 subjects × 10 images each)
    - Generated non-face images
    - Modified face images
    """
    # Initialize arrays for AT&T dataset
    n_subjects = 40
    images_per_subject = 10
    total_images = n_subjects * images_per_subject
    data = np.zeros((total_images, 10304))
    labels = np.zeros(total_images)

    # Load AT&T dataset
    count = 0
    for subject in range(1, n_subjects + 1):
        path = os.path.join(base_path, f's{subject}')
        for img_num in range(1, images_per_subject + 1):
            img_path = os.path.join(path, f'{img_num}.pgm')
            img = np.array(Image.open(img_path)).flatten()
            data[count] = img
            labels[count] = subject
            count += 1

    # Load non-face images
    non_face_images = []
    for filename in os.listdir(non_face_dir):
        if filename.endswith('.pgm'):
            img_path = os.path.join(non_face_dir, filename)
            img = np.array(Image.open(img_path)).flatten()
            non_face_images.append(img)

    # Load modified face images
    modified_images = []
    for filename in os.listdir(modified_face_dir):
        if filename.endswith('.pgm'):
            img_path = os.path.join(modified_face_dir, filename)
            img = np.array(Image.open(img_path)).flatten()
            modified_images.append(img)

    # Select test subjects (5 random subjects)
    all_subjects = list(range(1, n_subjects + 1))
    test_subject_ids = random.sample(all_subjects, 5)
    train_subject_ids = [x for x in all_subjects if x not in test_subject_ids]

    # Initialize lists for split indices
    train_indices = []
    test_indices_train_subjects = []
    test_indices_test_subjects = []

    # Split data based on subject IDs
    for subject in train_subject_ids:
        # Get all indices for this subject
        subject_indices = np.where(labels == subject)[0]
        # Randomly select 8 images for training
        train_images = random.sample(list(subject_indices), 8)
        # Use remaining 2 images for testing
        test_images = [
            idx for idx in subject_indices if idx not in train_images]

        train_indices.extend(train_images)
        test_indices_train_subjects.extend(test_images)

    # Add all images from test subjects to test set
    for subject in test_subject_ids:
        subject_indices = np.where(labels == subject)[0]
        test_indices_test_subjects.extend(subject_indices)

    # Create final datasets
    train_data = data[train_indices]
    train_labels = labels[train_indices]

    test_data_train = data[test_indices_train_subjects]
    test_labels_train = labels[test_indices_train_subjects]

    test_data_test = data[test_indices_test_subjects]
    test_labels_test = labels[test_indices_test_subjects]

    # Convert lists to arrays for additional test sets
    non_face_data = np.array(non_face_images)
    modified_face_data = np.array(modified_images)

    return (train_data, train_labels,
            test_data_train, test_labels_train,
            test_data_test, test_labels_test,
            non_face_data, modified_face_data,
            test_subject_ids)


class SimpleMultiScaleClassifier:
    def __init__(self, patch_sizes=[8, 16], distance_weight=10.0):
        self.patch_sizes = patch_sizes
        self.distance_weight = distance_weight
        self.class_means = {}
        self.classes = None

    def extract_gradient_features(self, image):
        """Simplified gradient feature extraction"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = convolve2d(image, sobel_x, mode='valid')
        grad_y = convolve2d(image, sobel_y, mode='valid')

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Simple histogram of gradient magnitudes
        hist, _ = np.histogram(magnitude, bins=16, density=True)
        return hist

    def extract_patch_features(self, image, patch_size):
        """Extract simple patch-based features"""
        height, width = image.shape
        features = []

        # Take mean of patches
        for i in range(0, height - patch_size + 1, patch_size):
            for j in range(0, width - patch_size + 1, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                features.append(np.mean(patch))

        return np.array(features)

    def extract_features(self, X):
        """Extract combined features"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        features_all = []

        for sample_idx in range(n_samples):
            image = X[sample_idx].reshape(112, 92)

            # Extract gradient features
            grad_features = self.extract_gradient_features(image)

            # Extract patch features at different scales
            patch_features = []
            for size in self.patch_sizes:
                patch_feat = self.extract_patch_features(image, size)
                patch_features.extend(patch_feat)

            # Combine all features
            combined_features = np.concatenate([grad_features, patch_features])
            features_all.append(combined_features)

        return np.array(features_all)

    def fit(self, X, y):
        """Fit the classifier"""
        self.classes = np.unique(y)
        features = self.extract_features(X)

        # Compute mean features for each class
        for c in self.classes:
            class_samples = features[y == c]
            self.class_means[c] = np.mean(class_samples, axis=0)

        return self

    def predict(self, X):
        """Predict class labels and confidences"""
        features = self.extract_features(X)
        predictions = []
        confidences = []

        for sample in features:
            distances = []
            for c in self.classes:
                dist = np.linalg.norm(sample - self.class_means[c])
                distances.append(dist)

            distances = np.array(distances)
            similarities = np.exp(-distances / self.distance_weight)
            conf_scores = similarities / (similarities.sum() + 1e-10)

            pred_idx = np.argmin(distances)
            predictions.append(self.classes[pred_idx])
            confidences.append(conf_scores[pred_idx])

        return np.array(predictions), np.array(confidences)


def evaluate_classification(train_data, train_labels,
                            test_data_train, test_labels_train,
                            test_data_test, test_labels_test,
                            non_face_data, modified_face_data):
    """Simplified evaluation function"""
    try:
        print("Training Simple Multi-scale Classifier...")
        clf = SimpleMultiScaleClassifier(patch_sizes=[8, 16])

        # Fit classifier
        clf.fit(train_data, train_labels)

        # Get predictions
        pred_train_subjects, conf_train = clf.predict(test_data_train)
        pred_test_subjects, conf_test = clf.predict(test_data_test)
        pred_non_face, conf_non_face = clf.predict(non_face_data)
        pred_modified, conf_modified = clf.predict(modified_face_data)

        results = {
            'identification': {
                'train_subjects': pred_train_subjects,
                'test_subjects': pred_test_subjects
            },
            'recognition': {
                'non_face': pred_non_face,
                'modified': pred_modified
            },
            'confidences': {
                'train': conf_train,
                'test': conf_test,
                'non_face': conf_non_face,
                'modified': conf_modified
            }
        }

        return results

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_results(results, test_labels_train, test_labels_test):
    """
    Print comprehensive evaluation results
    """
    print("\nFace Identification Results:")

    print("\nOn training subjects (2 images each):")
    acc_train = accuracy_score(
        test_labels_train,
        results['identification']['train_subjects']
    )
    print(f"Accuracy: {acc_train:.3f}")

    print("\nOn test subjects (10 images each):")
    acc_test = accuracy_score(
        test_labels_test,
        results['identification']['test_subjects']
    )
    print(f"Accuracy: {acc_test:.3f}")

    print("\nFace Recognition Results:")
    print("\nOn non-face images:")
    non_face_preds = results['recognition']['non_face']
    non_face_correct = np.mean(
        [pred not in test_labels_train for pred in non_face_preds]
    )
    print(f"Rejection rate: {non_face_correct:.3f}")

    print("\nOn modified face images:")
    modified_preds = results['recognition']['modified']
    modified_detection = np.mean(
        [pred in np.unique(test_labels_train) for pred in modified_preds]
    )
    print(f"Detection rate: {modified_detection:.3f}")

    print("\nConfidence Statistics:")
    for data_type in ['train', 'test', 'non_face', 'modified']:
        conf = results['confidences'][data_type]
        print(f"\n{data_type.capitalize()} images:")
        print(f"Mean confidence: {np.mean(conf):.3f}")
        print(f"Min confidence: {np.min(conf):.3f}")
        print(f"Max confidence: {np.max(conf):.3f}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load and split data
    print("Loading and splitting data...")
    (train_data, train_labels,
     test_data_train, test_labels_train,
     test_data_test, test_labels_test,
     non_face_data, modified_face_data,
     test_subject_ids) = load_and_split_data("./att_faces",
                                             "./non_face_images",
                                             "./modified_faces")

    # Print dataset information
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (training subjects): {len(test_data_train)} images")
    print(f"Test set (test subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    # Train classifier and evaluate
    print("\nTraining and evaluating Multi-scale Feature classifier...")
    results = evaluate_classification(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results
    print_results(results, test_labels_train, test_labels_test)


if __name__ == "__main__":
    main()
