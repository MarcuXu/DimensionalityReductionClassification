import numpy as np
from PIL import Image
import os
import random
import time


def load_and_split_data(base_path, non_face_dir, modified_face_dir):
    """
    Load and split the dataset
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


class CustomCNNClassifier:
    def __init__(self, learning_rate=0.001, epochs=20, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # CNN parameters
        self.conv1_filters = np.random.randn(16, 1, 3, 3) * 0.01
        self.conv1_bias = np.zeros(16)
        self.conv2_filters = np.random.randn(32, 16, 3, 3) * 0.01
        self.conv2_bias = np.zeros(32)

        self.fc1_weights = None
        self.fc1_bias = None
        self.fc2_weights = None
        self.fc2_bias = None

        self.classes = None

    def reshape_input(self, X):
        """Reshape input to proper format"""
        # If input is flattened (10304,), reshape to (112, 92)
        if len(X.shape) == 1:
            X = X.reshape(112, 92)

        # Resize to smaller dimension
        img = Image.fromarray(X.astype(np.uint8))
        img = img.resize((48, 48))
        X = np.array(img) / 255.0

        return X.reshape(1, 48, 48)

    def convolve2d(self, image, kernel):
        """2D convolution operation"""
        if len(image.shape) > 2:
            image = image.reshape(image.shape[1], image.shape[2])

        k_h, k_w = kernel.shape
        i_h, i_w = image.shape
        out_h = i_h - k_h + 1
        out_w = i_w - k_w + 1

        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                output[i, j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
        return output

    def forward_pass(self, x):
        batch_size = x.shape[0]
        # Reshape each image in the batch
        x_reshaped = np.zeros((batch_size, 1, 48, 48))
        for i in range(batch_size):
            x_reshaped[i, 0] = self.reshape_input(x[i]).reshape(48, 48)

        # First convolution layer
        conv1_out = np.zeros((batch_size, 16, 46, 46))
        for i in range(16):
            for j in range(batch_size):
                conv1_out[j, i] = self.convolve2d(
                    x_reshaped[j], self.conv1_filters[i, 0])
                conv1_out[j, i] += self.conv1_bias[i]
        conv1_out = np.maximum(0, conv1_out)  # ReLU

        # First max pooling
        pool1_out = np.zeros((batch_size, 16, 23, 23))
        for i in range(16):
            for j in range(batch_size):
                for h in range(23):
                    for w in range(23):
                        pool1_out[j, i, h, w] = np.max(
                            conv1_out[j, i, h*2:h*2+2, w*2:w*2+2])

        # Second convolution layer
        conv2_out = np.zeros((batch_size, 32, 21, 21))
        for i in range(32):
            for j in range(batch_size):
                for k in range(16):
                    conv2_out[j, i] += self.convolve2d(
                        pool1_out[j, k], self.conv2_filters[i, k])
                conv2_out[j, i] += self.conv2_bias[i]
        conv2_out = np.maximum(0, conv2_out)  # ReLU

        # Second max pooling
        pool2_out = np.zeros((batch_size, 32, 10, 10))
        for i in range(32):
            for j in range(batch_size):
                for h in range(10):
                    for w in range(10):
                        pool2_out[j, i, h, w] = np.max(
                            conv2_out[j, i, h*2:h*2+2, w*2:w*2+2])

        # Flatten and fully connected layers
        flattened = pool2_out.reshape(batch_size, -1)
        fc1 = np.maximum(
            0, np.dot(flattened, self.fc1_weights) + self.fc1_bias)
        logits = np.dot(fc1, self.fc2_weights) + self.fc2_bias

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilities

    def fit(self, X, y):
        print("Starting CNN training...")
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize FC layers
        feature_size = 32 * 10 * 10  # After conv and pool layers
        self.fc1_weights = np.random.randn(
            feature_size, 128) * np.sqrt(2.0 / feature_size)
        self.fc1_bias = np.zeros(128)
        self.fc2_weights = np.random.randn(128, n_classes) * np.sqrt(2.0 / 128)
        self.fc2_bias = np.zeros(n_classes)

        # Convert labels to one-hot
        y_onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            y_onehot[i, np.where(self.classes == label)[0]] = 1

        n_samples = len(X)
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0

            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)

                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y_onehot[batch_indices]

                predictions = self.forward_pass(X_batch)
                batch_loss = - \
                    np.mean(np.sum(y_batch * np.log(predictions + 1e-15), axis=1))
                total_loss += batch_loss

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.4f}")

    def predict(self, X):
        probabilities = self.forward_pass(X)
        predicted_classes = self.classes[np.argmax(probabilities, axis=1)]
        confidences = np.max(probabilities, axis=1)
        return predicted_classes, confidences

    def predict_proba(self, X):
        return self.forward_pass(X)


def evaluate_cnn_classification(train_data, train_labels,
                                test_data_train, test_labels_train,
                                test_data_test, test_labels_test,
                                non_face_data, modified_face_data):
    try:
        clf = CustomCNNClassifier(epochs=20, batch_size=32)
        print("Training CNN classifier...")
        clf.fit(train_data, train_labels)

        # Get predictions and confidences
        pred_train_subjects, conf_train = clf.predict(test_data_train)
        pred_test_subjects, conf_test = clf.predict(test_data_test)
        pred_non_face, conf_non_face = clf.predict(non_face_data)
        pred_modified, conf_modified = clf.predict(modified_face_data)

        # Compute accuracies
        train_acc = np.mean(pred_train_subjects == test_labels_train)
        test_acc = np.mean(pred_test_subjects == test_labels_test)

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
            'accuracies': {
                'train': train_acc,
                'test': test_acc
            }
        }
    except Exception as e:
        print(f"Error in CNN evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'identification': {'train_subjects': None, 'test_subjects': None},
            'recognition': {'non_face': None, 'modified': None},
            'confidences': {'train': None, 'test': None, 'non_face': None, 'modified': None},
            'accuracies': {'train': 0.0, 'test': 0.0}
        }


def print_results(results):
    """
    Print comprehensive results
    """
    print("\nFace Recognition Results (CNN):")

    print("\nOn training subjects:")
    acc_train = np.mean(results['confidences']['train'])
    print(f"Mean confidence: {acc_train:.3f}")
    print(f"Min confidence: {np.min(results['confidences']['train']):.3f}")
    print(f"Max confidence: {np.max(results['confidences']['train']):.3f}")

    print("\nOn test subjects:")
    acc_test = np.mean(results['confidences']['test'])
    print(f"Mean confidence: {acc_test:.3f}")
    print(f"Min confidence: {np.min(results['confidences']['test']):.3f}")
    print(f"Max confidence: {np.max(results['confidences']['test']):.3f}")

    print("\nOn non-face images:")
    acc_non_face = np.mean(results['confidences']['non_face'])
    print(f"Mean confidence: {acc_non_face:.3f}")
    print(f"Min confidence: {np.min(results['confidences']['non_face']):.3f}")
    print(f"Max confidence: {np.max(results['confidences']['non_face']):.3f}")

    print("\nOn modified face images:")
    acc_modified = np.mean(results['confidences']['modified'])
    print(f"Mean confidence: {acc_modified:.3f}")
    print(f"Min confidence: {np.min(results['confidences']['modified']):.3f}")
    print(f"Max confidence: {np.max(results['confidences']['modified']):.3f}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Set paths
    base_path = "./att_faces"
    non_face_dir = "./non_face_images"
    modified_face_dir = "./modified_faces"

    # Load all data
    print("Loading and splitting data...")
    (train_data, train_labels,
     test_data_train, test_labels_train,
     test_data_test, test_labels_test,
     non_face_data, modified_face_data,
     test_subject_ids) = load_and_split_data(base_path, non_face_dir, modified_face_dir)

    # Print dataset information
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} images")
    print(f"Test set (training subjects): {len(test_data_train)} images")
    print(f"Test set (test subjects): {len(test_data_test)} images")
    print(f"Non-face images: {len(non_face_data)} images")
    print(f"Modified face images: {len(modified_face_data)} images")
    print(f"Test subjects: {sorted(test_subject_ids)}")

    # Initialize and evaluate CNN classifier
    print("\nTraining and evaluating CNN classifier...")
    start_time = time.time()
    results = evaluate_cnn_classification(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )
    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
