import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path


class SoftmaxPCAClassifier:
    def __init__(self, n_components=50, learning_rate=0.01, n_iterations=500, batch_size=32):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.components = None
        self.mean = None
        self.feature_scale = None
        self.n_classes = 40  # For AT&T dataset
        self.weights = None
        self.bias = None
        self.explained_variance_ratio = None

    def standardize(self, X):
        """Standardize features"""
        return (X - self.mean) / (self.feature_scale + 1e-8)

    def softmax(self, X):
        """Compute softmax values for each set of scores in X"""
        # Subtract max for numerical stability
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """Compute cross entropy loss with L2 regularization"""
        m = y_true.shape[0]
        log_likelihood = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        l2_loss = 0.01 * np.sum(self.weights ** 2)  # L2 regularization
        return log_likelihood + l2_loss

    def _to_one_hot(self, y):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(y), self.n_classes))
        for i, label in enumerate(y):
            one_hot[i, int(label)-1] = 1
        return one_hot

    def fit(self, X, y):
        """Fit PCA and then train softmax classifier on reduced data"""
        # Compute mean and standard deviation for standardization
        self.mean = np.mean(X, axis=0)
        self.feature_scale = np.std(X, axis=0) + 1e-8
        X_standardized = self.standardize(X)

        # Perform PCA using SVD
        U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)

        # Calculate explained variance ratio
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_var = explained_variance.sum()
        self.explained_variance_ratio = explained_variance / total_var

        # Store principal components
        self.components = Vt[:self.n_components].T

        # Project data onto PCA space
        X_pca = np.dot(X_standardized, self.components)

        # Initialize weights and bias
        self.weights = np.random.randn(
            self.n_components, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)

        # Convert labels to one-hot
        y_one_hot = self._to_one_hot(y)

        # Initialize best parameters for early stopping
        best_loss = float('inf')
        best_weights = None
        best_bias = None
        patience = 5
        patience_counter = 0

        # Mini-batch gradient descent
        n_samples = X_pca.shape[0]

        for epoch in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_pca[indices]
            y_shuffled = y_one_hot[indices]

            total_loss = 0

            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward pass
                logits = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.softmax(logits)

                # Compute gradients
                error = y_pred - y_batch
                dw = (1/len(X_batch)) * np.dot(X_batch.T,
                                               error) + 0.01 * self.weights
                db = (1/len(X_batch)) * np.sum(error, axis=0)

                # Gradient clipping
                clip_value = 5.0
                dw = np.clip(dw, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Compute loss
                batch_loss = self.cross_entropy_loss(y_pred, y_batch)
                total_loss += batch_loss

            # Early stopping check
            avg_loss = total_loss / (n_samples // self.batch_size)
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

    def project(self, X):
        """Project data onto PCA space"""
        X_standardized = self.standardize(X)
        return np.dot(X_standardized, self.components)

    def predict(self, X):
        """Predict classes and provide confidence scores"""
        X_proj = self.project(X)
        logits = np.dot(X_proj, self.weights) + self.bias
        probas = self.softmax(logits)

        predictions = np.argmax(probas, axis=1) + 1
        confidences = np.max(probas, axis=1)

        return predictions, confidences, probas


def evaluate_classification(clf, train_data, train_labels,
                            test_data_train, test_labels_train,
                            test_data_test, test_labels_test,
                            non_face_data, modified_face_data):
    """Evaluate both face recognition and identification"""
    # Train the classifier
    clf.fit(train_data, train_labels)

    # Get predictions for all datasets
    train_pred, train_conf, train_prob = clf.predict(train_data)
    pred_train_subjects, conf_train, prob_train = clf.predict(test_data_train)
    pred_test_subjects, conf_test, prob_test = clf.predict(test_data_test)
    pred_non_face, conf_non_face, prob_non_face = clf.predict(non_face_data)
    pred_modified, conf_modified, prob_modified = clf.predict(
        modified_face_data)

    # Calculate metrics
    train_acc = np.mean(train_pred == train_labels)
    test_acc_train = np.mean(pred_train_subjects == test_labels_train)
    test_acc_test = np.mean(pred_test_subjects == test_labels_test)

    # Face recognition metrics
    non_face_rejection = np.mean([conf < 0.5 for conf in conf_non_face])
    modified_detection = np.mean([conf >= 0.5 for conf in conf_modified])

    results = {
        'accuracies': {
            'train': train_acc,
            'test_known': test_acc_train,
            'test_unknown': test_acc_test
        },
        'recognition': {
            'non_face_rejection': non_face_rejection,
            'modified_detection': modified_detection
        },
        'confidences': {
            'train': train_conf,
            'test_known': conf_train,
            'test_unknown': conf_test,
            'non_face': conf_non_face,
            'modified': conf_modified
        }
    }

    return results


def load_and_split_data(base_path, non_face_dir, modified_face_dir):
    """
    Load and split the data for face recognition and identification
    """
    # Initialize arrays for AT&T dataset
    n_subjects = 40
    images_per_subject = 10
    total_images = n_subjects * images_per_subject
    data = np.zeros((total_images, 10304))  # 112*92 = 10304
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

    # Split data
    train_indices = []
    test_indices_train_subjects = []
    test_indices_test_subjects = []

    for subject in train_subject_ids:
        subject_indices = np.where(labels == subject)[0]
        train_images = random.sample(list(subject_indices), 8)
        test_images = [
            idx for idx in subject_indices if idx not in train_images]

        train_indices.extend(train_images)
        test_indices_train_subjects.extend(test_images)

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

    # Convert additional test sets to arrays
    non_face_data = np.array(non_face_images)
    modified_face_data = np.array(modified_images)

    return (train_data, train_labels,
            test_data_train, test_labels_train,
            test_data_test, test_labels_test,
            non_face_data, modified_face_data,
            test_subject_ids)


def print_evaluation_results(results):
    """
    Print comprehensive evaluation results
    """
    print("\nClassification Results:")
    print(f"Training Accuracy: {results['accuracies']['train']:.3f}")
    print(
        f"Test Accuracy (Known Subjects): {results['accuracies']['test_known']:.3f}")
    print(
        f"Test Accuracy (Unknown Subjects): {results['accuracies']['test_unknown']:.3f}")

    # Calculate and print overall test accuracy
    n_known = len(results['confidences']['test_known'])
    n_unknown = len(results['confidences']['test_unknown'])
    overall_acc = (results['accuracies']['test_known'] * n_known +
                   results['accuracies']['test_unknown'] * n_unknown) / (n_known + n_unknown)
    print(f"Overall Test Accuracy: {overall_acc:.3f}")

    print("\nFace Recognition Results:")
    print(
        f"Non-face Image Rejection Rate: {results['recognition']['non_face_rejection']:.3f}")
    print(
        f"Modified Face Detection Rate: {results['recognition']['modified_detection']:.3f}")

    print("\nConfidence Statistics:")
    for data_type in ['train', 'test_known', 'test_unknown', 'non_face', 'modified']:
        confidences = results['confidences'][data_type]
        print(f"\n{data_type.replace('_', ' ').title()} Images:")
        print(f"Mean confidence: {np.mean(confidences):.3f}")
        print(f"Min confidence: {np.min(confidences):.3f}")
        print(f"Max confidence: {np.max(confidences):.3f}")


def visualize_predictions(clf, samples, true_labels, sample_type, output_dir):
    """
    Visualize predictions for sample images with confidence scores and true labels.

    Parameters:
    -----------
    clf : SoftmaxPCAClassifier
        Trained classifier
    samples : array
        Sample images to visualize
    true_labels : array or None
        True labels (None for non-face images)
    sample_type : str
        Type of samples ('training', 'test_known', 'test_unknown', 'non_face', 'modified')
    output_dir : str
        Directory to save visualization
    """
    n_samples = min(5, len(samples))  # Show up to 5 samples
    indices = np.random.choice(len(samples), n_samples, replace=False)

    plt.figure(figsize=(15, 3*n_samples))

    for idx, sample_idx in enumerate(indices):
        # Get predictions
        pred, conf, probs = clf.predict(samples[sample_idx:sample_idx+1])
        pred = pred[0]
        conf = conf[0]

        # Plot image
        plt.subplot(n_samples, 2, 2*idx + 1)
        plt.imshow(samples[sample_idx].reshape(112, 92), cmap='gray')
        plt.axis('off')

        # Create title based on sample type
        if sample_type == 'non_face':
            title = f'Prediction: {"Non-Face" if conf < 0.5 else f"Face (Subject {pred})"}\n'
            title += f'Confidence: {conf:.3f}'
        else:
            if true_labels is not None:
                true_label = true_labels[sample_idx]
                title = f'True: Subject {int(true_label)} | Pred: Subject {pred}\n'
            else:
                title = f'Prediction: Subject {pred}\n'
            title += f'Confidence: {conf:.3f}'

        plt.title(title)

        # Plot probability distribution
        plt.subplot(n_samples, 2, 2*idx + 2)
        top_k = 5  # Show top-k predictions
        top_indices = np.argsort(probs[0])[-top_k:][::-1]
        top_probs = probs[0][top_indices]

        bars = plt.bar(range(top_k), top_probs)
        plt.xlabel('Subject ID')
        plt.ylabel('Probability')
        plt.title('Top-5 Prediction Probabilities')

        # Add subject IDs and probability values on bars
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            plt.text(i, prob, f'Sub {idx+1}\n{prob:.3f}',
                     ha='center', va='bottom')

            # Highlight the true label bar if available
            if true_labels is not None and idx + 1 == true_labels[sample_idx]:
                bars[i].set_color('green')
            elif true_labels is None and conf < 0.5 and sample_type == 'non_face':
                bars[i].set_color('red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_{sample_type}.png'))
    plt.close()


def visualize_all_predictions(clf, train_data, train_labels,
                              test_data_train, test_labels_train,
                              test_data_test, test_labels_test,
                              non_face_data, modified_face_data,
                              output_dir):
    """
    Generate visualizations for all types of predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Visualize training samples
    visualize_predictions(clf, train_data, train_labels,
                          'training', output_dir)

    # Visualize test samples (known subjects)
    visualize_predictions(clf, test_data_train, test_labels_train,
                          'test_known', output_dir)

    # Visualize test samples (unknown subjects)
    visualize_predictions(clf, test_data_test, test_labels_test,
                          'test_unknown', output_dir)

    # Visualize non-face samples
    visualize_predictions(clf, non_face_data, None,
                          'non_face', output_dir)

    # Visualize modified face samples
    visualize_predictions(clf, modified_face_data, None,
                          'modified', output_dir)


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create output directories
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Load and split data
    print("Loading and splitting data...")
    (train_data, train_labels,
     test_data_train, test_labels_train,
     test_data_test, test_labels_test,
     non_face_data, modified_face_data,
     test_subject_ids) = load_and_split_data(
        "./att_faces",
        "./non_face_images",
        "./modified_faces"
    )

    # Print dataset information
    print("\nDataset Information:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (known subjects): {len(test_data_train)} images")
    print(f"Test set (unknown subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    print("\nTraining Softmax PCA classifier...")
    clf = SoftmaxPCAClassifier(n_components=50)

    # Evaluate the classifier
    results = evaluate_classification(
        clf, train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results
    print_evaluation_results(results)

    # Generate prediction visualizations
    print("\nGenerating prediction visualizations...")
    visualize_all_predictions(
        clf, train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data,
        'output'
    )

    print("\nVisualization files saved in 'output' directory")


if __name__ == "__main__":
    main()
