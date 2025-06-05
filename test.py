import kagglehub

# Download latest version
path = kagglehub.dataset_download("hmendonca/cloud-cover-detection")

print("Path to dataset files:", path)