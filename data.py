import os
import kagglehub
from roboflow import Roboflow

# Define the common download directory
download_dir = r"C:\Users\acer\SMF"

# Step 1: Download the Kaggle dataset (Face Mask Detection) to the specified directory
def download_kaggle_dataset():
    path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    # Move the dataset to the specified directory
    new_path = os.path.join(download_dir, os.path.basename(path))
    os.rename(path, new_path)  # Moves the dataset to the target directory
    print("Kaggle Dataset downloaded to:", new_path)
    return new_path

# Step 2: Download the Roboflow dataset (Hairnet Dataset) to the specified directory
def download_roboflow_dataset():
    rf = Roboflow(api_key="VN28ceooZnimGyMKzkNo")
    project = rf.workspace("agts").project("hairnet")
    version = project.version(15)
    dataset = version.download("yolov5")  # Download in YOLOv5 format

    # Move the dataset to the specified directory
    new_path = os.path.join(download_dir, "hairnet-15")
    os.rename(dataset.location, new_path)  # Moves the dataset to the target directory
    print("Roboflow Dataset downloaded to:", new_path)
    return new_path

# Main function to download both datasets
def main():
    # Download datasets
    kaggle_path = download_kaggle_dataset()
    roboflow_path = download_roboflow_dataset()

    print("Both datasets have been downloaded.")

if __name__ == '__main__':
    main()
