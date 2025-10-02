import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import utils.ext_transforms as et
from datasets.cityscapes import Cityscapes
from finetune_nobatchnorm_lightning import DeepLabModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics.stream_metrics import StreamSegMetrics

def create_train_id_to_raw_id_map():
    """
    Create a mapping from train_ids (0-18) back to raw ids based on Cityscapes class definitions
    """
    # Based on the Cityscapes classes in cityscapes.py
    # Find the raw_id for each train_id
    train_id_to_raw_id = {}
    
    for cls in Cityscapes.classes:
        raw_id = cls.id
        train_id = cls.train_id
        
        # Skip license plate which has id -1
        if raw_id >= 0:
            if train_id not in train_id_to_raw_id:
                train_id_to_raw_id[train_id] = raw_id
    
    # Map the ignore class (255) to raw id 0 (unlabeled)
    train_id_to_raw_id[255] = 0
    
    return train_id_to_raw_id

def calculate_metrics_with_valid_labels(test_dataset, predictions_list, num_classes=34):
    """
    Calculate metrics for images where we have valid labels (not all 255)
    
    Args:
        test_dataset: The test dataset
        predictions_list: List of raw_id predictions
        num_classes: Number of classes in raw id space
    
    Returns:
        Dictionary with metrics or None if no valid labels found
    """
    # Create metrics calculator
    metrics = StreamSegMetrics(num_classes)
    
    valid_count = 0
    
    for i, (pred, data_item) in enumerate(zip(predictions_list, test_dataset)):
        # Handle both tensor and numpy array targets
        # When using Subset, the target might already be a numpy array
        if isinstance(data_item, tuple):
            _, target = data_item
            if isinstance(target, torch.Tensor):
                target_np = target.cpu().numpy()
            else:
                target_np = target
        else:
            # If data_item is not a tuple, it might be the target itself
            if isinstance(data_item, torch.Tensor):
                target_np = data_item.cpu().numpy()
            else:
                target_np = data_item
        
        valid_mask = (target_np != 255)
        
        if valid_mask.sum() > 0:
            valid_count += 1
            metrics.update(target_np[valid_mask], pred[valid_mask])
    
    if valid_count == 0:
        print("No valid labels found in any test images!")
        return None
    
    print(f"Found {valid_count} images with valid labels for evaluation")
    return metrics.get_results()

def map_predictions_to_raw_ids(model_path, output_dir, num_samples=None):
    """
    Load model, make predictions on test set, and map predictions back to raw label IDs
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save visualizations
        num_samples: Number of samples to process (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = DeepLabModel.load_from_checkpoint(model_path)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Create the train_id to raw_id mapping
    train_id_to_raw_id_map = create_train_id_to_raw_id_map()
    print(f"Created mapping from train_ids to raw_ids: {train_id_to_raw_id_map}")
    
    # Setup transforms
    test_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the test dataset
    test_dataset = Cityscapes(
        root='/nfs/shared/cityscapes',
        split='test',
        transform=test_transforms
    )
    
    if num_samples is not None:
        subset_indices = list(range(min(num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Process one at a time for visualization
        shuffle=False,
        num_workers=2
    )
    
    print(f"Processing {len(test_loader)} test images")
    
    # Keep track of all predictions for metrics
    all_raw_predictions = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            # Get model prediction
            images = images.to(device)  # Move images to the same device as model
            outputs = model(images)
            predictions = outputs.max(dim=1)[1].cpu().numpy()
            
            # Map predictions from train_ids back to raw_ids
            raw_predictions = np.zeros_like(predictions)
            for train_id, raw_id in train_id_to_raw_id_map.items():
                if train_id == 255:  # Skip ignore class for mapping
                    continue
                raw_predictions[predictions == train_id] = raw_id
            
            # Save prediction for metrics calculation
            all_raw_predictions.append(raw_predictions[0])
            
            # Get the original images and targets
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8)
            
            # The targets are already in raw_id format for test set
            raw_target = targets[0].cpu().numpy()
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Create a masked array for ground truth to show ignored pixels as transparent
            plt.subplot(1, 3, 2)
            masked_gt = np.ma.masked_where(raw_target == 255, raw_target)
            plt.imshow(img)  # Show image underneath
            plt.imshow(masked_gt, vmin=0, vmax=33, cmap='tab20', alpha=0.7)
            plt.title(f"Ground Truth (Raw IDs)\nUnique: {np.unique(raw_target)}")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(raw_predictions[0], vmin=0, vmax=33, cmap='tab20')
            plt.title(f"Prediction (Mapped to Raw IDs)\nUnique: {np.unique(raw_predictions[0])}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"test_prediction_{i}.png"))
            plt.close()
            
            # Stop after processing specified number of samples
            if num_samples is not None and i >= num_samples - 1:
                break
    
    print(f"Saved visualizations to {output_dir}")
    
    # Calculate and print metrics if we have valid labels
    print("\nAttempting to calculate metrics...")
    metrics = calculate_metrics_with_valid_labels(test_dataset, all_raw_predictions)
    
    if metrics:
        print("\n===== Test Metrics (Raw ID Space) =====")
        print(f"Mean IoU: {metrics['Mean IoU']:.4f}")
        print(f"Overall Acc: {metrics['Overall Acc']:.4f}")
        print(f"Mean Acc: {metrics['Mean Acc']:.4f}")
        
        # Print IoU per class 
        print("\nIoU per class:")
        for class_id, iou in metrics['Class IoU'].items():
            # Only show classes that have predictions
            if not np.isnan(iou) and iou > 0:
                class_name = next((cls.name for cls in Cityscapes.classes if cls.id == class_id), f"Unknown ({class_id})")
                print(f"  Class {class_id} ({class_name}): {iou:.4f}")

if __name__ == "__main__":
    model_path = "/nfs/homedirs/duk/DeepLabV3Plus-Pytorch/checkpoints_no_batchnorm/deeplabv3_no_bn_crop_512_best_miou.ckpt"
    output_dir = "/nfs/homedirs/duk/DeepLabV3Plus-Pytorch/test_predictions"
    
    # Process 5 test images
    map_predictions_to_raw_ids(model_path, output_dir, num_samples=5) 