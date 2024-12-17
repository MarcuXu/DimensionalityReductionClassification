import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score between true labels and predicted labels"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


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
        subject_indices = np.where(labels == subject)[0]
        train_images = random.sample(list(subject_indices), 8)
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


class TwoStageLinearClassifier:
    def __init__(self, n_components=50, learning_rate=0.0001, n_iterations=1000):  # Reduced learning rate
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.components = None
        self.mean = None
        self.feature_scale = None
        self.face_detector_weights = None
        self.face_detector_bias = None
        self.identifier_weights = None
        self.identifier_bias = None
        self.n_classes = 40
        self.detection_threshold = 0.2

    def standardize(self, X):
        """Standardize the features"""
        return (X - self.mean) / (self.feature_scale + 1e-8)

    def _safe_exp(self, x):
        """Compute exponential in a numerically stable way"""
        # Clip values to prevent overflow
        # np.log(np.finfo(np.float64).max) ≈ 709
        return np.exp(np.clip(x, -709, 709))

    def _train_face_detector(self, face_data_pca, non_face_data_pca):
        """Train binary classifier for face detection with balanced weights"""
        X_detector = np.vstack([face_data_pca, non_face_data_pca])
        y_detector = np.hstack(
            [np.ones(len(face_data_pca)), np.zeros(len(non_face_data_pca))])

        # Initialize with smaller random values
        self.face_detector_weights = np.random.randn(self.n_components) * 0.001
        self.face_detector_bias = 0

        # Add weights to balance the classes
        n_faces = len(face_data_pca)
        n_non_faces = len(non_face_data_pca)
        weights = np.ones(len(y_detector))
        weights[y_detector == 0] *= (n_faces / n_non_faces)

        best_weights = self.face_detector_weights.copy()
        best_bias = self.face_detector_bias
        best_accuracy = 0
        v_w = np.zeros_like(self.face_detector_weights)
        v_b = 0
        beta = 0.9

        for iteration in range(self.n_iterations):
            # Forward pass with clipping
            scores = np.clip(
                np.dot(X_detector, self.face_detector_weights) +
                self.face_detector_bias,
                -100, 100
            )
            predictions = scores > self.detection_threshold

            current_accuracy = np.mean(predictions == y_detector)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weights = self.face_detector_weights.copy()
                best_bias = self.face_detector_bias

            # Compute gradients with clipping
            error = np.clip(scores - y_detector, -100, 100)
            dw = np.clip((2/len(X_detector)) *
                         np.dot(X_detector.T, error * weights), -1, 1)
            db = np.clip((2/len(X_detector)) * np.sum(error * weights), -1, 1)

            # Update with momentum and gradient clipping
            v_w = beta * v_w - self.learning_rate * dw
            v_b = beta * v_b - self.learning_rate * db

            # Clip updates
            v_w = np.clip(v_w, -1, 1)
            v_b = np.clip(v_b, -1, 1)

            self.face_detector_weights += v_w
            self.face_detector_bias += v_b

        self.face_detector_weights = best_weights
        self.face_detector_bias = best_bias

    def _train_identifier(self, X_pca, y):
        """Train face identifier"""
        self.identifier_weights = np.random.randn(
            self.n_components, self.n_classes) * 0.001
        self.identifier_bias = np.zeros(self.n_classes)

        y_one_hot = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            y_one_hot[i, int(label)-1] = 1

        best_weights = self.identifier_weights.copy()
        best_bias = self.identifier_bias.copy()
        best_loss = float('inf')
        v_w = np.zeros_like(self.identifier_weights)
        v_b = np.zeros_like(self.identifier_bias)
        beta = 0.9

        for iteration in range(self.n_iterations):
            # Forward pass with clipping
            scores = np.clip(
                np.dot(X_pca, self.identifier_weights) + self.identifier_bias,
                -100, 100
            )

            # Compute loss with clipping
            current_loss = np.mean(np.clip((scores - y_one_hot) ** 2, 0, 100))

            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = self.identifier_weights.copy()
                best_bias = self.identifier_bias.copy()

            # Compute gradients with clipping
            error = np.clip(scores - y_one_hot, -100, 100)
            dw = np.clip((2/len(X_pca)) * np.dot(X_pca.T, error), -1, 1)
            db = np.clip((2/len(X_pca)) * np.sum(error, axis=0), -1, 1)

            # Add L2 regularization
            dw += 0.01 * self.identifier_weights

            # Update with momentum and gradient clipping
            v_w = beta * v_w - self.learning_rate * dw
            v_b = beta * v_b - self.learning_rate * db

            # Clip updates
            v_w = np.clip(v_w, -1, 1)
            v_b = np.clip(v_b, -1, 1)

            self.identifier_weights += v_w
            self.identifier_bias += v_b

        self.identifier_weights = best_weights
        self.identifier_bias = best_bias

    def fit(self, X, y, non_face_data):
        """
        Fit both face detector and identifier

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training face images
        y : array-like of shape (n_samples,)
            Labels for face images
        non_face_data : array-like of shape (n_non_face_samples, n_features)
            Training non-face images
        """
        try:
            print("Starting model fitting...")

            # Compute mean and standard deviation for standardization
            self.mean = np.mean(X, axis=0)
            self.feature_scale = np.std(X, axis=0) + 1e-8

            # Standardize both face and non-face data
            print("Standardizing data...")
            X_standardized = self.standardize(X)
            non_face_standardized = self.standardize(non_face_data)

            # Compute PCA
            print("Computing PCA...")
            U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)
            self.components = Vt[:self.n_components].T

            # Project data to PCA space
            print("Projecting data to PCA space...")
            X_pca = np.dot(X_standardized, self.components)
            non_face_pca = np.dot(non_face_standardized, self.components)

            # Train face detector
            print("Training face detector...")
            self._train_face_detector(X_pca, non_face_pca)

            # Train face identifier
            print("Training face identifier...")
            self._train_identifier(X_pca, y)

            print("Model fitting completed successfully")
            return self

        except Exception as e:
            print(f"Error during model fitting: {str(e)}")
            raise

    def predict(self, X):
        """Two-stage prediction with numerical stability"""
        X_standardized = self.standardize(X)
        X_pca = np.dot(X_standardized, self.components)

        # Face detection with clipping
        face_scores = np.clip(
            np.dot(X_pca, self.face_detector_weights) +
            self.face_detector_bias,
            -100, 100
        )
        is_face = face_scores > self.detection_threshold

        # Face identification with clipping
        id_scores = np.clip(
            np.dot(X_pca, self.identifier_weights) + self.identifier_bias,
            -100, 100
        )
        predictions = np.argmax(id_scores, axis=1) + 1

        predictions[~is_face] = -1

        # Calculate confidences with numerical stability
        face_scores_stable = np.clip(
            face_scores - self.detection_threshold, -100, 100)
        face_confidences = 1 / (1 + self._safe_exp(-2 * face_scores_stable))
        id_confidences = np.max(self._softmax(id_scores), axis=1)
        confidences = face_confidences * id_confidences

        return predictions, confidences

    def _softmax(self, x):
        """Compute softmax values with improved numerical stability"""
        # Shift values to prevent overflow
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = self._safe_exp(shifted_x)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)

    def predict_proba(self, X):
        """Predict probabilities with numerical stability"""
        X_standardized = self.standardize(X)
        X_pca = np.dot(X_standardized, self.components)
        scores = np.clip(
            np.dot(X_pca, self.identifier_weights) + self.identifier_bias,
            -100, 100
        )
        return self._softmax(scores)


def evaluate_classification_two_stage(train_data, train_labels,
                                      test_data_train, test_labels_train,
                                      test_data_test, test_labels_test,
                                      non_face_data, modified_face_data,
                                      n_components=50):
    """Evaluate using two-stage classifier"""
    try:
        # Initialize and train classifier
        print("Initializing classifier...")
        clf = TwoStageLinearClassifier(n_components=n_components)

        print("Fitting classifier...")
        clf.fit(train_data, train_labels, non_face_data)

        print("Getting predictions...")
        # Get predictions for training data
        train_pred, train_conf = clf.predict(train_data)

        # Get predictions for test sets
        pred_train_subjects, conf_train = clf.predict(test_data_train)
        pred_test_subjects, conf_test = clf.predict(test_data_test)
        pred_non_face, conf_non_face = clf.predict(non_face_data)
        pred_modified, conf_modified = clf.predict(modified_face_data)

        # Get prediction probabilities
        prob_train = clf.predict_proba(test_data_train)
        prob_test = clf.predict_proba(test_data_test)
        prob_non_face = clf.predict_proba(non_face_data)
        prob_modified = clf.predict_proba(modified_face_data)

        print("Evaluation completed successfully")
        return {
            'training': {
                'predictions': train_pred,
                'labels': train_labels
            },
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
            }
        }

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_results(results, test_labels_train, test_labels_test):
    """Print comprehensive evaluation results"""
    print("\nTraining Set Results:")
    # Get valid indices for training predictions (where prediction != -1)
    train_pred = results['training']['predictions']
    train_labels = results['training']['labels']
    valid_train = train_pred != -1
    if np.any(valid_train):  # Only calculate if there are valid predictions
        train_acc = accuracy_score(
            train_labels[valid_train],
            train_pred[valid_train]
        )
        print(f"Training Accuracy: {train_acc:.3f}")
    else:
        print("Training Accuracy: No valid predictions")

    print("\nFace Identification Results:")

    print("\nOn training subjects (2 images each):")
    train_subj_pred = results['identification']['train_subjects']
    valid_train_subj = train_subj_pred != -1
    if np.any(valid_train_subj):
        acc_train = accuracy_score(
            test_labels_train[valid_train_subj],
            train_subj_pred[valid_train_subj]
        )
        print(f"Accuracy: {acc_train:.3f}")
    else:
        print("Accuracy: No valid predictions")

    print("\nOn test subjects (10 images each):")
    test_subj_pred = results['identification']['test_subjects']
    valid_test_subj = test_subj_pred != -1
    if np.any(valid_test_subj):
        acc_test = accuracy_score(
            test_labels_test[valid_test_subj],
            test_subj_pred[valid_test_subj]
        )
        print(f"Accuracy: {acc_test:.3f}")
    else:
        print("Accuracy: No valid predictions")

    # Calculate overall test accuracy
    all_test_predictions = np.concatenate([
        results['identification']['train_subjects'],
        results['identification']['test_subjects']
    ])
    all_test_labels = np.concatenate([test_labels_train, test_labels_test])
    valid_indices = all_test_predictions != -1

    if np.any(valid_indices):
        overall_test_acc = accuracy_score(
            all_test_labels[valid_indices],
            all_test_predictions[valid_indices]
        )
        print(
            f"\nOverall Test Accuracy (all test images): {overall_test_acc:.3f}")
    else:
        print("\nOverall Test Accuracy: No valid predictions")

    print("\nFace Recognition Results:")

    print("\nOn non-face images:")
    non_face_preds = results['recognition']['non_face']
    non_face_correct = np.mean(non_face_preds == -1)
    print(f"Rejection rate: {non_face_correct:.3f}")

    print("\nOn modified face images:")
    modified_preds = results['recognition']['modified']
    modified_detection = np.mean(modified_preds != -1)
    print(f"Detection rate: {modified_detection:.3f}")

    print("\nConfidence Statistics:")
    for data_type in ['train', 'test', 'non_face', 'modified']:
        conf = results['confidences'][data_type]
        print(f"\n{data_type.capitalize()} images:")
        if len(conf) > 0:
            print(f"Mean confidence: {np.mean(conf):.3f}")
            print(f"Min confidence: {np.min(conf):.3f}")
            print(f"Max confidence: {np.max(conf):.3f}")
        else:
            print("No confidence scores available")


def visualize_eigenfaces(clf, num_components=5):
    """Visualize the first few PCA components (eigenfaces)"""
    plt.figure(figsize=(15, 3))
    for i in range(num_components):
        plt.subplot(1, num_components, i + 1)
        eigenface = clf.components[:, i].reshape(112, 92)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Component {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('eigenfaces_two_stage.png')
    plt.close()


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
     test_subject_ids) = load_and_split_data(
        "./att_faces", "./non_face_images", "./modified_faces")

    # Print dataset information
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (training subjects): {len(test_data_train)} images")
    print(f"Test set (test subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    # Train and evaluate two-stage classifier
    print("\nTraining and evaluating two-stage classifier...")
    results = evaluate_classification_two_stage(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results and generate visualizations
    print_results(results, test_labels_train, test_labels_test)

    # Visualize eigenfaces
    clf = TwoStageLinearClassifier()
    clf.fit(train_data, train_labels, non_face_data)
    visualize_eigenfaces(clf)


if __name__ == "__main__":
    main()
