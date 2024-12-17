import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_face_data(base_path):
    """
    Load face images from the AT&T database
    Returns: Data matrix (400 x 10304) and labels
    """
    # Initialize data matrices
    data = np.zeros((400, 10304))  # 10304 = 112 x 92 pixels
    labels = np.zeros(400)

    count = 0
    # Loop through all 40 subjects
    for subject in range(1, 41):
        path = os.path.join(base_path, f's{subject}')
        # Load 10 images for each subject
        for img_num in range(1, 11):
            img_path = os.path.join(path, f'{img_num}.pgm')
            # Read and flatten image
            img = np.array(Image.open(img_path)).flatten()
            data[count] = img
            labels[count] = subject
            count += 1

    return data, labels


def compute_eigenfaces(data):
    """
    Compute eigenfaces using PCA
    Returns: eigenfaces, mean_face, explained_variance_ratio
    """
    # Get mean face
    mean_face = np.mean(data, axis=0)

    # Center the data
    centered_data = data - mean_face

    # Compute the covariance matrix (using the trick for high dimensional data)
    # Instead of computing X X^T (10304x10304), compute X^T X (400x400)
    covariance_matrix = np.dot(centered_data, centered_data.T)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert eigenvectors to eigenfaces
    # Project centered data onto eigenvectors to get eigenfaces
    eigenfaces = np.dot(centered_data.T, eigenvectors)

    # Normalize eigenfaces
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    return eigenfaces, mean_face, explained_variance_ratio


def plot_eigenfaces(eigenfaces, n_faces=5):
    """
    Plot the first n_faces eigenfaces
    """
    fig, axes = plt.subplots(1, n_faces, figsize=(15, 3))
    for i in range(n_faces):
        # Correctly reshape the eigenface
        eigenface = eigenfaces[:, i].reshape(112, 92)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Eigenface {i+1}')
    plt.tight_layout()
    return fig


def plot_variance_explained(explained_variance_ratio):
    """
    Plot cumulative explained variance ratio
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    return plt.gcf()


def reconstruct_face(face, eigenfaces, mean_face, n_components):
    """
    Reconstruct a face using n_components eigenfaces
    """
    # Center the face
    centered_face = face - mean_face

    # Project onto eigenspace
    weights = np.dot(centered_face, eigenfaces[:, :n_components])

    # Reconstruct
    reconstruction = mean_face + np.dot(eigenfaces[:, :n_components], weights)

    return reconstruction.reshape(112, 92)


def main(data_path):
    try:
        # Load data
        print("Loading face data...")
        data, labels = load_face_data(data_path)

        # Compute eigenfaces
        print("Computing eigenfaces...")
        eigenfaces, mean_face, explained_variance_ratio = compute_eigenfaces(
            data)

        # Question 1(a): Visualize leading eigenfaces
        print("Plotting first 5 eigenfaces...")
        fig_eigenfaces = plot_eigenfaces(eigenfaces)
        fig_eigenfaces.savefig('eigenfaces.png')
        print("Eigenfaces plot saved")

        # Question 1(b): Analyze eigenface importance
        print("Plotting explained variance...")
        fig_variance = plot_variance_explained(explained_variance_ratio)
        fig_variance.savefig('variance_explained.png')
        print("Variance plot saved")

        # Example reconstruction with different numbers of components
        test_face = data[0]  # First face in dataset
        components_to_test = [5, 10, 25, 50, 100, 200, 300]

        fig, axes = plt.subplots(
            1, len(components_to_test) + 1, figsize=(15, 3))
        axes[0].imshow(test_face.reshape(112, 92), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i, n_comp in enumerate(components_to_test):
            reconstruction = reconstruct_face(
                test_face, eigenfaces, mean_face, n_comp)
            axes[i+1].imshow(reconstruction, cmap='gray')
            axes[i+1].set_title(f'{n_comp} components')
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.savefig('reconstructions.png')
        print("Reconstruction examples saved")

        print("Process completed successfully!")
        return eigenfaces, mean_face, explained_variance_ratio

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Replace with your dataset path
    data_path = "./att_faces"  # Update this to your actual path
    eigenfaces, mean_face, explained_variance_ratio = main(data_path)
