import os
import random
import nibabel as nib

# Dataset paths
image_folder = r"C:\Medical_segment\New folder\CervixRawData\RawData\Training\img"
label_folder = r"C:\Medical_segment\New folder\CervixRawData\RawData\Training\label"

# Get all image files
image_files = sorted(os.listdir(image_folder))

# Randomly choose 5 images
random_images = random.sample(image_files, 5)

print("="*70)
print("Randomly Selected Image-Label Pairs")
print("="*70)

for image_file in random_images:

    # Generate corresponding label filename
    label_file = image_file.replace("-Image.nii", "-Mask.nii")

    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, label_file)

    # Load image and label
    image = nib.load(image_path)
    label = nib.load(label_path)

    image_data = image.get_fdata()
    label_data = label.get_fdata()

    print(f"\nImage : {image_file}")
    print(f"Label : {label_file}")
    print(f"Image Shape : {image_data.shape}")
    print(f"Label Shape : {label_data.shape}")
    print(f"Unique Labels : {set(label_data.flatten())}")