import numpy as np
from PIL import Image
import os
import random
from scipy.ndimage import rotate, zoom
import cv2


def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_non_face_images(output_dir, num_images=10):
    """
    Generate non-face images using different methods:
    1. Random noise
    2. Geometric patterns
    3. Gradient patterns
    """
    create_directory(output_dir)

    # Image dimensions to match AT&T face database
    width, height = 92, 112

    for i in range(num_images):
        if i < 3:  # Random noise
            # Generate random noise
            noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            img = Image.fromarray(noise)
            img.save(os.path.join(output_dir, f'noise_{i+1}.pgm'))

        elif i < 6:  # Geometric patterns
            # Create a blank image
            img = np.zeros((height, width), dtype=np.uint8)

            # Draw random shapes
            for _ in range(5):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                thickness = random.randint(1, 3)
                color = random.randint(100, 255)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # Add some rectangles
            for _ in range(3):
                x = random.randint(0, width-20)
                y = random.randint(0, height-20)
                w = random.randint(10, 30)
                h = random.randint(10, 30)
                color = random.randint(100, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(output_dir, f'geometric_{i-2}.pgm'))

        else:  # Gradient patterns
            # Create gradient
            x = np.linspace(0, 255, width)
            y = np.linspace(0, 255, height)
            xx, yy = np.meshgrid(x, y)
            # Convert to float32 for calculations
            gradient = ((xx + yy) % 256).astype(np.float32)

            # Add random variations (now using float32)
            variations = np.random.uniform(-30, 30,
                                           gradient.shape).astype(np.float32)
            gradient += variations

            # Clip and convert to uint8
            gradient = np.clip(gradient, 0, 255)
            gradient = gradient.astype(np.uint8)

            img = Image.fromarray(gradient)
            img.save(os.path.join(output_dir, f'gradient_{i-5}.pgm'))


def generate_modified_faces(input_dir, output_dir, num_variations=5):
    """
    Generate modified versions of existing face images by applying:
    1. Rotation
    2. Scaling
    3. Noise addition
    4. Brightness/contrast changes
    """
    create_directory(output_dir)

    # Get random face images
    subject_dirs = [d for d in os.listdir(input_dir) if d.startswith('s')]
    selected_subject = random.choice(subject_dirs)
    subject_path = os.path.join(input_dir, selected_subject)

    # Load a random image from the selected subject
    image_files = [f for f in os.listdir(subject_path) if f.endswith('.pgm')]
    source_image = Image.open(os.path.join(
        subject_path, random.choice(image_files)))
    # Convert to float32 for calculations
    source_array = np.array(source_image, dtype=np.float32)

    for i in range(num_variations):
        if i == 0:  # Rotation
            angle = random.uniform(-15, 15)
            modified = rotate(source_array, angle, reshape=False)

        elif i == 1:  # Scaling
            scale = random.uniform(0.8, 1.2)
            modified = zoom(source_array, scale, order=3)
            # Ensure correct size
            if modified.shape != (112, 92):
                modified = cv2.resize(modified, (92, 112))

        elif i == 2:  # Add noise
            noise = np.random.normal(
                0, 25, source_array.shape).astype(np.float32)
            modified = source_array + noise

        elif i == 3:  # Brightness change
            factor = random.uniform(0.7, 1.3)
            modified = source_array * factor

        else:  # Contrast change
            factor = random.uniform(0.5, 1.5)
            modified = ((source_array - 128) * factor) + 128

        # Clip and convert to uint8
        modified = np.clip(modified, 0, 255)
        modified = modified.astype(np.uint8)

        # Save modified image
        img = Image.fromarray(modified)
        img.save(os.path.join(output_dir, f'modified_face_{i+1}.pgm'))


def main():
    try:
        # Create output directories
        non_face_dir = './non_face_images'
        modified_face_dir = './modified_faces'

        # Generate test images
        print("Generating non-face images...")
        generate_non_face_images(non_face_dir)

        print("Generating modified face images...")
        generate_modified_faces('./att_faces', modified_face_dir)

        print("Done! Generated images are saved in:")
        print(f"- Non-face images: {non_face_dir}")
        print(f"- Modified faces: {modified_face_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
