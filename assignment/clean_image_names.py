import os
import time

# Rename images in all directories

def rename_images(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter out non-image files (optional)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Rename each image file
    for i, filename in enumerate(image_files):
        try:
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]
            
            # Create the new filename
            new_filename = f"{i + 1}{file_extension}"
            
            # Construct the full old and new file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")
            
            # Add a small delay to avoid overwhelming the file system
            time.sleep(0.01)
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

# Set the base directory
base_dir = os.getcwd()  # Get the current working directory

# Construct the path to the dataset
dataset_dir = os.path.join(base_dir, 'dataset')

# Construct the path to the training directory
train_dir = os.path.join(dataset_dir, 'train')

# Construct the path to the validation directory
val_dir = os.path.join(dataset_dir, 'val')

# Construct the path to the test directory
test_dir = os.path.join(dataset_dir, 'test')

# Rename images in each training subdirectory
# rename_images(os.path.join(train_dir, 'headtop'))
# rename_images(os.path.join(train_dir, 'helmet'))
# rename_images(os.path.join(train_dir, 'hoodie'))
rename_images(os.path.join(train_dir, 'no_headwear'))

# Rename images in each validation subdirectory
# rename_images(os.path.join(val_dir, 'headtop'))
# rename_images(os.path.join(val_dir, 'helmet'))
# rename_images(os.path.join(val_dir, 'hoodie'))
rename_images(os.path.join(val_dir, 'no_headwear'))

# Rename images in each test subdirectory
# rename_images(os.path.join(test_dir, 'headtop'))
# rename_images(os.path.join(test_dir, 'helmet'))
# rename_images(os.path.join(test_dir, 'hoodie'))
rename_images(os.path.join(test_dir, 'no_headwear'))