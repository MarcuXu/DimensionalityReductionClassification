import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_and_split_data(base_path, non_face_dir, modified_face_dir):
    """
    Load all data including:
    - Original AT&T dataset
    - Generated non-face images
    - Modified face images
    """
    # Initialize arrays for AT&T dataset
    n_subjects = 40
    n_train_subjects = 35
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

    # Split AT&T dataset
    all_subjects = list(range(1, n_subjects + 1))
    test_subject_ids = random.sample(all_subjects, 5)
    train_subject_ids = [x for x in all_subjects if x not in test_subject_ids]

    # Split data
    train_indices = []
    test_indices_train_subjects = []
    test_indices_test_subjects = []

    for i, label in enumerate(labels):
        if label in train_subject_ids:
            subject_images = list(range(i - (i % 10), i - (i % 10) + 10))
            train_images = random.sample(subject_images, 8)
            test_images = [x for x in subject_images if x not in train_images]
            train_indices.extend(train_images)
            test_indices_train_subjects.extend(test_images)
        else:
            test_indices_test_subjects.append(i)

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


class CustomPCAClassifier:
    def __init__(self, n_components=50, distance_weight=10.0):
        self.n_components = n_components
        self.distance_weight = distance_weight  # Weight for distance scaling
        self.components = None
        self.mean = None
        self.class_means = {}
        self.class_covariances = {}  # Store class covariances
        self.classes = None
        self.explained_variance_ratio = None
        self.training_features = None
        self.training_labels = None
        self.feature_std = None  # Store feature standard deviation

    def fit(self, X, y):
        """
        Fit PCA and compute class statistics in reduced space
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Calculate explained variance ratio
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_var = explained_variance.sum()
        self.explained_variance_ratio = explained_variance / total_var

        # Store principal components
        self.components = Vt[:self.n_components].T

        # Project training data
        self.training_features = np.dot(X_centered, self.components)

        # Normalize features
        self.feature_std = np.std(self.training_features, axis=0)
        self.feature_std[self.feature_std == 0] = 1
        self.training_features = self.training_features / self.feature_std

        self.training_labels = y
        self.classes = np.unique(y)

        # Compute class statistics
        for c in self.classes:
            class_samples = self.training_features[y == c]
            self.class_means[c] = np.mean(class_samples, axis=0)
            # Compute class covariance with regularization
            cov = np.cov(class_samples.T)
            # Add small constant to diagonal for stability
            cov += np.eye(cov.shape[0]) * 1e-6
            self.class_covariances[c] = cov

    def project(self, X):
        """
        Project data onto normalized PCA space
        """
        X_centered = X - self.mean
        X_proj = np.dot(X_centered, self.components)
        return X_proj / self.feature_std

    def mahalanobis_distance(self, sample, class_mean, class_cov):
        """
        Compute Mahalanobis distance between sample and class
        """
        diff = sample - class_mean
        try:
            inv_cov = np.linalg.inv(class_cov)
            dist = np.sqrt(diff.dot(inv_cov).dot(diff))
            return dist
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if inversion fails
            return np.linalg.norm(diff)

    def compute_confidence(self, distances):
        """
        Compute confidence score based on distances
        """
        # Convert distances to similarities using exponential
        similarities = np.exp(-distances / self.distance_weight)

        # Normalize similarities to [0, 1]
        if similarities.sum() > 0:
            return similarities / similarities.sum()
        return similarities

    def predict(self, X):
        """
        Predict classes and provide confidence scores
        """
        X_proj = self.project(X)
        predictions = []
        confidences = []

        for sample in X_proj:
            # Compute distances to all classes using Mahalanobis distance
            distances = np.array([
                self.mahalanobis_distance(
                    sample, self.class_means[c], self.class_covariances[c])
                for c in self.classes
            ])

            # Get prediction and confidence
            min_dist_idx = np.argmin(distances)
            predicted_class = self.classes[min_dist_idx]

            # Compute confidence scores
            conf_scores = self.compute_confidence(distances)
            confidence = conf_scores[min_dist_idx]

            predictions.append(predicted_class)
            confidences.append(confidence)

        return np.array(predictions), np.array(confidences)

    def predict_proba(self, X):
        """
        Predict probability-like scores for each class
        """
        X_proj = self.project(X)
        probas = []

        for sample in X_proj:
            # Compute distances to all classes
            distances = np.array([
                self.mahalanobis_distance(
                    sample, self.class_means[c], self.class_covariances[c])
                for c in self.classes
            ])

            # Convert distances to probabilities
            probas.append(self.compute_confidence(distances))

        return np.array(probas)


def evaluate_classification(train_data, train_labels,
                            test_data_train, test_labels_train,
                            test_data_test, test_labels_test,
                            non_face_data, modified_face_data,
                            n_components=50):
    """
    Evaluate both face recognition and identification using improved PCA classifier
    """
    try:
        # Initialize and train classifier
        clf = CustomPCAClassifier(
            n_components=n_components, distance_weight=15.0)
        clf.fit(train_data, train_labels)

        # Get predictions
        pred_train_subjects, conf_train = clf.predict(test_data_train)
        pred_test_subjects, conf_test = clf.predict(test_data_test)
        pred_non_face, conf_non_face = clf.predict(non_face_data)
        pred_modified, conf_modified = clf.predict(modified_face_data)

        # Get prediction probabilities
        prob_train = clf.predict_proba(test_data_train)
        prob_test = clf.predict_proba(test_data_test)
        prob_non_face = clf.predict_proba(non_face_data)
        prob_modified = clf.predict_proba(modified_face_data)

        return {
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
            },
            'probabilities': {
                'train': prob_train,
                'test': prob_test,
                'non_face': prob_non_face,
                'modified': prob_modified
            },
            'explained_variance_ratio': clf.explained_variance_ratio
        }
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# class CustomPCAClassifier:
#     def __init__(self, n_components=50, threshold=0.8):
#         self.n_components = n_components
#         self.threshold = threshold
#         self.components = None
#         self.mean = None
#         self.class_means = {}
#         self.classes = None
#         self.explained_variance_ratio = None
#         self.training_features = None
#         self.training_labels = None

#     def fit(self, X, y):
#         """
#         Fit PCA and compute class means in reduced space
#         """
#         # Center the data
#         self.mean = np.mean(X, axis=0)
#         X_centered = X - self.mean

#         # Compute SVD
#         U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

#         # Calculate explained variance ratio
#         explained_variance = (s ** 2) / (X.shape[0] - 1)
#         total_var = explained_variance.sum()
#         self.explained_variance_ratio = explained_variance / total_var

#         # Store principal components
#         self.components = Vt[:self.n_components].T

#         # Project training data
#         self.training_features = np.dot(X_centered, self.components)
#         self.training_labels = y
#         self.classes = np.unique(y)

#         # Compute mean feature vector for each class
#         for c in self.classes:
#             class_samples = self.training_features[y == c]
#             self.class_means[c] = np.mean(class_samples, axis=0)

#     def project(self, X):
#         """
#         Project data onto PCA space
#         """
#         X_centered = X - self.mean
#         return np.dot(X_centered, self.components)

#     def stable_softmax(self, x):
#         """
#         Compute softmax values in a numerically stable way
#         """
#         # Subtract the maximum value for numerical stability
#         shifted_x = x - np.max(x)
#         exp_x = np.exp(shifted_x)
#         return exp_x / np.sum(exp_x)

#     def compute_distance_scores(self, sample):
#         """
#         Compute distance scores to all class means
#         """
#         distances = []
#         classes = []
#         for c in sorted(self.class_means.keys()):
#             dist = np.linalg.norm(sample - self.class_means[c])
#             distances.append(dist)
#             classes.append(c)
#         return np.array(distances), np.array(classes)

#     def predict(self, X):
#         """
#         Predict classes and provide confidence scores
#         """
#         X_proj = self.project(X)
#         predictions = []
#         confidences = []

#         for sample in X_proj:
#             distances, classes = self.compute_distance_scores(sample)

#             # Find minimum distance and corresponding class
#             min_dist_idx = np.argmin(distances)
#             predicted_class = classes[min_dist_idx]

#             # Compute confidence score using scaled exponential
#             confidence = np.exp(-distances[min_dist_idx] / self.threshold)
#             # Clip confidence to [0, 1]
#             confidence = np.clip(confidence, 0, 1)

#             predictions.append(predicted_class)
#             confidences.append(confidence)

#         return np.array(predictions), np.array(confidences)

#     def predict_proba(self, X):
#         """
#         Predict probability-like scores for each class
#         """
#         X_proj = self.project(X)
#         probas = []

#         for sample in X_proj:
#             distances, _ = self.compute_distance_scores(sample)

#             # Convert distances to similarity scores (negative distances)
#             similarity_scores = -distances / self.threshold

#             # Apply stable softmax
#             proba = self.stable_softmax(similarity_scores)
#             probas.append(proba)

#         return np.array(probas)


# def evaluate_classification(train_data, train_labels,
#                             test_data_train, test_labels_train,
#                             test_data_test, test_labels_test,
#                             non_face_data, modified_face_data,
#                             n_components=50):
#     """
#     Evaluate both face recognition and identification using custom PCA classifier
#     """
#     try:
#         # Initialize and train classifier
#         clf = CustomPCAClassifier(n_components=n_components)
#         clf.fit(train_data, train_labels)

#         # Get predictions
#         pred_train_subjects, conf_train = clf.predict(test_data_train)
#         pred_test_subjects, conf_test = clf.predict(test_data_test)
#         pred_non_face, conf_non_face = clf.predict(non_face_data)
#         pred_modified, conf_modified = clf.predict(modified_face_data)

#         # Get prediction probabilities
#         prob_train = clf.predict_proba(test_data_train)
#         prob_test = clf.predict_proba(test_data_test)
#         prob_non_face = clf.predict_proba(non_face_data)
#         prob_modified = clf.predict_proba(modified_face_data)

#         return {
#             'identification': {
#                 'train_subjects': pred_train_subjects,
#                 'test_subjects': pred_test_subjects
#             },
#             'recognition': {
#                 'non_face': pred_non_face,
#                 'modified': pred_modified
#             },
#             'confidences': {
#                 'train': conf_train,
#                 'test': conf_test,
#                 'non_face': conf_non_face,
#                 'modified': conf_modified
#             },
#             'probabilities': {
#                 'train': prob_train,
#                 'test': prob_test,
#                 'non_face': prob_non_face,
#                 'modified': prob_modified
#             },
#             'explained_variance_ratio': clf.explained_variance_ratio
#         }
#     except Exception as e:
#         print(f"Error in evaluation: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None


def visualize_eigenfaces(clf, num_components=5):
    """
    Visualize the first few PCA components (eigenfaces)
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_components):
        plt.subplot(1, num_components, i + 1)
        eigenface = clf.components[:, i].reshape(112, 92)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Component {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('eigenfaces_new.png')
    plt.close()


def print_results(results, test_labels_train, test_labels_test):
    """
    Print comprehensive evaluation results
    """
    print("\nFace Identification Results:")

    print("\nOn training subjects (2 images each):")
    acc_train = accuracy_score(
        test_labels_train, results['identification']['train_subjects'])
    print(f"Accuracy: {acc_train:.3f}")

    print("\nOn test subjects (10 images each):")
    acc_test = accuracy_score(
        test_labels_test, results['identification']['test_subjects'])
    print(f"Accuracy: {acc_test:.3f}")

    print("\nFace Recognition Results:")

    print("\nOn non-face images:")
    # Check if predictions are not in training subject IDs (should be rejected)
    non_face_preds = results['recognition']['non_face']
    non_face_correct = np.mean(
        [pred not in test_labels_train for pred in non_face_preds])
    print(f"Rejection rate: {non_face_correct:.3f}")

    print("\nOn modified face images:")
    # Check if predictions are in training subject IDs (should be accepted as faces)
    modified_preds = results['recognition']['modified']
    modified_detection = np.mean(
        [pred in np.unique(test_labels_train) for pred in modified_preds])
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

    # Load and split data (same as before)
    print("Loading and splitting data...")
    (train_data, train_labels,
     test_data_train, test_labels_train,
     test_data_test, test_labels_test,
     non_face_data, modified_face_data,
     test_subject_ids) = load_and_split_data("./att_faces", "./non_face_images", "./modified_faces")

    # Print dataset information
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (training subjects): {len(test_data_train)} images")
    print(f"Test set (test subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    # Train classifier and evaluate
    print("\nTraining and evaluating custom PCA classifier...")
    results = evaluate_classification(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results and generate visualizations
    print_results(results, test_labels_train, test_labels_test)

    # Visualize eigenfaces
    clf = CustomPCAClassifier()
    clf.fit(train_data, train_labels)
    visualize_eigenfaces(clf)


if __name__ == "__main__":
    main()
