import os
import argparse
import subprocess
import sys

def download_model(output_path, model_name="best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"):
    """Download DeepLabV3+ pretrained model weights using gdown.
    
    Args:
        output_path: Output directory to save the model
        model_name: Name to save the model file as
    """
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # File ID on Google Drive
    # This is the ID for the DeepLabV3+ ResNet101 Cityscapes model
    file_id = "17JLIsZPvLKMNHwOrEgk5-mSWlZqcwYEf"
    
    # Output file path
    output_file = os.path.join(output_path, model_name)
    
    # Download the file using gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model from {url} to {output_file}...")
    gdown.download(url, output_file, quiet=False)
    
    # Verify file was downloaded successfully
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
        print(f"Successfully downloaded model file ({file_size:.2f} MB)")
    else:
        print("Failed to download model file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DeepLabV3+ pretrained model weights")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Output directory to save the model")
    parser.add_argument("--model_name", type=str, 
                        default="best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar",
                        help="Filename to save the model as")
    
    args = parser.parse_args()
    download_model(args.output_dir, args.model_name) 