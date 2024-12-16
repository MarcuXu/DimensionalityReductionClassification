import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import seaborn as sns


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


def compute_eigenfaces(data, n_components=50):
    """
    Compute eigenfaces using PCA
    """
    mean_face = np.mean(data, axis=0)
    centered_data = data - mean_face
    covariance_matrix = np.dot(centered_data, centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenfaces = np.dot(centered_data.T, eigenvectors)

    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

    if n_components is not None:
        eigenfaces = eigenfaces[:, :n_components]

    return eigenfaces, mean_face


def project_faces(data, eigenfaces, mean_face):
    """
    Project faces onto eigenspace
    """
    centered_data = data - mean_face
    return np.dot(centered_data, eigenfaces)


def evaluate_classification(train_data, train_labels,
                            test_data_train, test_labels_train,
                            test_data_test, test_labels_test,
                            non_face_data, modified_face_data,
                            n_components=50):
    """
    Evaluate both face recognition and identification
    """
    # Compute eigenfaces
    eigenfaces, mean_face = compute_eigenfaces(train_data, n_components)

    # Project all data
    train_projected = project_faces(train_data, eigenfaces, mean_face)
    test_projected_train = project_faces(
        test_data_train, eigenfaces, mean_face)
    test_projected_test = project_faces(test_data_test, eigenfaces, mean_face)
    non_face_projected = project_faces(non_face_data, eigenfaces, mean_face)
    modified_face_projected = project_faces(
        modified_face_data, eigenfaces, mean_face)

    # Train classifier
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(train_projected, train_labels)

    # Get predictions
    pred_train_subjects = clf.predict(test_projected_train)
    pred_test_subjects = clf.predict(test_projected_test)
    pred_non_face = clf.predict(non_face_projected)
    pred_modified = clf.predict(modified_face_projected)

    # Get prediction probabilities
    prob_train = clf.predict_proba(test_projected_train)
    prob_test = clf.predict_proba(test_projected_test)
    prob_non_face = clf.predict_proba(non_face_projected)
    prob_modified = clf.predict_proba(modified_face_projected)

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
            'train': np.max(prob_train, axis=1),
            'test': np.max(prob_test, axis=1),
            'non_face': np.max(prob_non_face, axis=1),
            'modified': np.max(prob_modified, axis=1)
        }
    }


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


def visualize_results(results, test_data_train, test_labels_train,
                      test_data_test, test_labels_test,
                      non_face_data, modified_face_data,
                      test_subject_ids):
    """
    Create comprehensive visualizations of the results
    """

    # Create a figure with subplots
    plt.figure(figsize=(15, 10))

    # 1. Identification Results Visualization
    plt.subplot(2, 2, 1)
    # Compute confusion matrix for training subjects
    cm_train = confusion_matrix(test_labels_train,
                                results['identification']['train_subjects'])
    sns.heatmap(cm_train, cmap='Blues', annot=True, fmt='d', cbar=False)
    plt.title('Confusion Matrix - Training Subjects')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 2. Recognition Confidence Distribution
    plt.subplot(2, 2, 2)
    plt.hist([results['confidences']['train'],
             results['confidences']['non_face'],
             results['confidences']['modified']],
             label=['Training Faces', 'Non-faces', 'Modified Faces'],
             bins=20, alpha=0.7)
    plt.title('Recognition Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()

    # 3. Example Predictions
    plt.subplot(2, 2, 3)
    visualize_example_predictions(results, test_data_train, test_labels_train,
                                  non_face_data, modified_face_data)

    # 4. Performance Metrics
    plt.subplot(2, 2, 4)
    plot_performance_metrics(results, test_labels_train, test_labels_test)

    plt.tight_layout()
    plt.savefig('classification_results_svc.png')
    plt.close()

    # Create separate visualization for sample images and their predictions
    visualize_sample_predictions(results, test_data_train, test_data_test,
                                 non_face_data, modified_face_data)


def visualize_example_predictions(results, test_data_train, test_labels_train,
                                  non_face_data, modified_face_data):
    """
    Visualize example predictions for each category
    """
    categories = ['Training', 'Non-face', 'Modified']
    accuracies = [
        np.mean(results['identification']
                ['train_subjects'] == test_labels_train),
        np.mean(
            [pred not in test_labels_train for pred in results['recognition']['non_face']]),
        np.mean([pred in np.unique(test_labels_train)
                for pred in results['recognition']['modified']])
    ]

    plt.bar(categories, accuracies)
    plt.title('Recognition Performance by Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')


def plot_performance_metrics(results, test_labels_train, test_labels_test):
    """
    Plot various performance metrics
    """
    metrics = {
        'Train Acc': accuracy_score(test_labels_train,
                                    results['identification']['train_subjects']),
        'Test Acc': accuracy_score(test_labels_test,
                                   results['identification']['test_subjects']),
        'Non-face\nRejection': np.mean([pred not in test_labels_train
                                        for pred in results['recognition']['non_face']]),
        'Modified\nDetection': np.mean([pred in np.unique(test_labels_train)
                                        for pred in results['recognition']['modified']])
    }

    plt.bar(metrics.keys(), metrics.values())
    plt.title('Performance Metrics Overview')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')


def visualize_sample_predictions(results, test_data_train, test_data_test,
                                 non_face_data, modified_face_data):
    """
    Create a grid of sample images with their predictions
    """
    plt.figure(figsize=(15, 12))

    # Function to reshape flattened image data
    def reshape_image(img):
        return img.reshape(112, 92)

    # 1. Sample training subject predictions
    for i in range(4):
        plt.subplot(4, 4, i+1)
        plt.imshow(reshape_image(test_data_train[i]), cmap='gray')
        pred = results['identification']['train_subjects'][i]
        conf = results['confidences']['train'][i]
        plt.title(f'Train Pred: {pred}\nConf: {conf:.2f}')
        plt.axis('off')

    # 2. Sample non-face predictions
    for i in range(4):
        plt.subplot(4, 4, i+5)
        plt.imshow(reshape_image(non_face_data[i]), cmap='gray')
        pred = results['recognition']['non_face'][i]
        conf = results['confidences']['non_face'][i]
        plt.title(f'Non-face Pred: {pred}\nConf: {conf:.2f}')
        plt.axis('off')

    # 3. Sample modified face predictions
    for i in range(4):
        plt.subplot(4, 4, i+9)
        plt.imshow(reshape_image(modified_face_data[i]), cmap='gray')
        pred = results['recognition']['modified'][i]
        conf = results['confidences']['modified'][i]
        plt.title(f'Modified Pred: {pred}\nConf: {conf:.2f}')
        plt.axis('off')

    # 4. Sample test subject predictions
    for i in range(4):
        plt.subplot(4, 4, i+13)
        plt.imshow(reshape_image(test_data_test[i]), cmap='gray')
        pred = results['identification']['test_subjects'][i]
        conf = results['confidences']['test'][i]
        plt.title(f'Test Pred: {pred}\nConf: {conf:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions_svc.png')
    plt.close()


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

    # Evaluate classification
    print("\nEvaluating classification...")
    results = evaluate_classification(
        train_data, train_labels,
        test_data_train, test_labels_train,
        test_data_test, test_labels_test,
        non_face_data, modified_face_data
    )

    # Print results
    print_results(results, test_labels_train, test_labels_test)
    print("\nGenerating visualizations...")
    visualize_results(results,
                      test_data_train, test_labels_train,
                      test_data_test, test_labels_test,
                      non_face_data, modified_face_data,
                      test_subject_ids)


if __name__ == "__main__":
    main()
