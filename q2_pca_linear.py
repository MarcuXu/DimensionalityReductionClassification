import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score between true labels and predicted labels.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    float
        Accuracy score between 0.0 and 1.0

    Example:
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1, 2, 3, 5, 4]
    >>> custom_accuracy_score(y_true, y_pred)
    0.6
    """
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if arrays have same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")

    # Calculate number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Calculate accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy


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


class LinearRegressionPCAClassifier:
    def __init__(self, n_components=50, learning_rate=0.0001, n_iterations=500):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.components = None
        self.mean = None
        self.weights = None
        self.bias = None
        self.explained_variance_ratio = None
        self.n_classes = 40  # Fixed number of classes for AT&T dataset
        self.feature_scale = None

    def standardize(self, X):
        """Standardize the features"""
        return (X - self.mean) / (self.feature_scale + 1e-8)

    def fit(self, X, y):
        """
        Fit PCA and then train linear regression on reduced data
        """
        # Compute mean and standard deviation for standardization
        self.mean = np.mean(X, axis=0)
        self.feature_scale = np.std(X, axis=0) + 1e-8
        X_standardized = self.standardize(X)

        # Compute SVD
        U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)

        # Calculate explained variance ratio
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_var = explained_variance.sum()
        self.explained_variance_ratio = explained_variance / total_var

        # Store principal components
        self.components = Vt[:self.n_components].T

        # Project training data to PCA space
        X_pca = np.dot(X_standardized, self.components)

        # Initialize weights and bias with small random values
        self.weights = np.random.randn(
            self.n_components, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)

        # Convert labels to one-hot encoding
        y_one_hot = self._to_one_hot(y)

        # Mini-batch gradient descent
        batch_size = 32
        n_samples = X_pca.shape[0]
        best_loss = float('inf')
        best_weights = None
        best_bias = None
        patience = 5
        patience_counter = 0

        for epoch in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_pca[indices]
            y_shuffled = y_one_hot[indices]

            total_loss = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self._forward(X_batch)

                # Compute gradients with L2 regularization
                error = y_pred - y_batch
                dw = (1/len(X_batch)) * np.dot(X_batch.T,
                                               error) + 0.01 * self.weights
                db = (1/len(X_batch)) * np.sum(error, axis=0)

                # Gradient clipping
                dw = np.clip(dw, -1, 1)
                db = np.clip(db, -1, 1)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Accumulate loss
                total_loss += np.mean(error**2)

            # Early stopping check
            avg_loss = total_loss / (n_samples // batch_size)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.weights = best_weights
                self.bias = best_bias
                break

    def _to_one_hot(self, y):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            one_hot[i, int(label)-1] = 1
        return one_hot

    def _forward(self, X):
        """Forward pass of linear regression"""
        return np.dot(X, self.weights) + self.bias

    def project(self, X):
        """Project data onto PCA space"""
        X_standardized = self.standardize(X)
        return np.dot(X_standardized, self.components)

    def predict(self, X):
        """
        Predict classes and provide confidence scores
        """
        X_proj = self.project(X)
        scores = self._forward(X_proj)

        # Softmax for better probability estimation
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        predictions = np.argmax(scores, axis=1) + 1
        confidences = np.max(probas, axis=1)

        return predictions, confidences

    def predict_proba(self, X):
        """
        Predict probability-like scores for each class
        """
        X_proj = self.project(X)
        scores = self._forward(X_proj)

        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probas


def evaluate_classification_linear(train_data, train_labels,
                                   test_data_train, test_labels_train,
                                   test_data_test, test_labels_test,
                                   non_face_data, modified_face_data,
                                   n_components=50):
    """
    Evaluate both face recognition and identification using linear regression classifier
    """
    try:
        # Initialize and train classifier
        clf = LinearRegressionPCAClassifier(n_components=n_components)
        clf.fit(train_data, train_labels)

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
            },
            'explained_variance_ratio': clf.explained_variance_ratio
        }
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_results(results, test_labels_train, test_labels_test):
    """
    Print comprehensive evaluation results
    """
    print("\nTraining Set Results:")
    train_acc = accuracy_score(
        results['training']['labels'],
        results['training']['predictions']
    )
    print(f"Training Accuracy: {train_acc:.3f}")

    print("\nFace Identification Results:")

    print("\nOn training subjects (2 images each):")
    acc_train = accuracy_score(
        test_labels_train, results['identification']['train_subjects'])
    print(f"Accuracy: {acc_train:.3f}")

    print("\nOn test subjects (10 images each):")
    acc_test = accuracy_score(
        test_labels_test, results['identification']['test_subjects'])
    print(f"Accuracy: {acc_test:.3f}")

    # Calculate overall test accuracy
    all_test_predictions = np.concatenate([
        results['identification']['train_subjects'],
        results['identification']['test_subjects']
    ])
    all_test_labels = np.concatenate([test_labels_train, test_labels_test])
    overall_test_acc = accuracy_score(all_test_labels, all_test_predictions)
    print(f"\nOverall Test Accuracy (all test images): {overall_test_acc:.3f}")

    print("\nFace Recognition Results:")

    print("\nOn non-face images:")
    non_face_preds = results['recognition']['non_face']
    non_face_correct = np.mean(
        [pred not in test_labels_train for pred in non_face_preds])
    print(f"Rejection rate: {non_face_correct:.3f}")

    print("\nOn modified face images:")
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
    plt.savefig('eigenfaces_linear.png')
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
     test_subject_ids) = load_and_split_data("./att_faces", "./non_face_images", "./modified_faces")

    # Print dataset information
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (training subjects): {len(test_data_train)} images")
    print(f"Test set (test subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    # Train and evaluate linear regression classifier
    print("\nTraining and evaluating linear regression PCA classifier...")
    results = evaluate_classification_linear(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results and generate visualizations
    print_results(results, test_labels_train, test_labels_test)

    # Visualize eigenfaces
    clf = LinearRegressionPCAClassifier()
    clf.fit(train_data, train_labels)
    visualize_eigenfaces(clf)


if __name__ == "__main__":
    main()
