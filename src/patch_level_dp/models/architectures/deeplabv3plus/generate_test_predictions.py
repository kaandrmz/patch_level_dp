import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import utils.ext_transforms as et
from datasets.cityscapes import Cityscapes
from finetune_nobatchnorm_lightning import DeepLabModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    
    # Dataset options
    parser.add_argument("--data_root", type=str, default='/nfs/shared/cityscapes',
                        help="Path to the Cityscapes dataset")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Image crop size")
    
    # Model options
    parser.add_argument("--ckpt", type=str, 
                        default="/nfs/homedirs/anon/DeepLabV3Plus-Pytorch/checkpoints_no_batchnorm/deeplabv3_no_bn_crop_512_best_miou.ckpt",
                        help="Path to model checkpoint")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="/nfs/homedirs/anon/DeepLabV3Plus-Pytorch/test_predictions",
                        help="Output directory for predictions")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of test samples to process")
    
    return parser

def main():
    opts = get_argparser().parse_args()
    
    # Create output directory
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # Determine device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {opts.ckpt}")
    model = DeepLabModel.load_from_checkpoint(opts.ckpt)
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Setup transforms for test images
    test_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load Cityscapes test dataset
    test_dataset = Cityscapes(
        root=opts.data_root,
        split='test',
        transform=test_transform
    )
    
    # Take a subset if specified
    if opts.num_samples is not None and opts.num_samples > 0:
        subset_indices = list(range(min(opts.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Processing {len(test_loader)} test images")
    
    # Create a mapping from train_id to original id
    train_id_to_id = {}
    for cls in Cityscapes.classes:
        if cls.train_id != 255 and cls.train_id != -1:  # Skip ignored classes
            train_id_to_id[cls.train_id] = cls.id
    
    # Colormap for visualizing predictions in Cityscapes colors
    color_map = []
    for cls in Cityscapes.classes:
        color_map.append(cls.color)
    
    # Process test images
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader)):
            # Get model prediction
            images = images.to(device)
            outputs = model(images)
            preds = outputs.max(dim=1)[1].cpu().numpy()
            
            # Get original image for visualization
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8)
            
            # Create colored segmentation mask
            colored_mask = np.zeros((preds.shape[1], preds.shape[2], 3), dtype=np.uint8)
            for train_id in range(19):  # 19 Cityscapes classes
                mask = preds[0] == train_id
                if mask.sum() > 0:
                    orig_id = train_id_to_id[train_id]
                    colored_mask[mask] = color_map[orig_id]
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(colored_mask)
            plt.title(f"Prediction\nUnique classes: {np.unique(preds[0])}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(opts.output_dir, f"test_prediction_{i}.png"))
            plt.close()
    
    print(f"Predictions saved to {opts.output_dir}")
    
    # Also run prediction on validation set to demonstrate metrics
    print("\nCalculating metrics on validation set...")
    
    val_dataset = Cityscapes(
        root=opts.data_root,
        split='val',
        transform=test_transform
    )
    
    # Take a subset if specified
    if opts.num_samples is not None and opts.num_samples > 0:
        subset_indices = list(range(min(opts.num_samples, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Reset model's test evaluator
    model.test_evaluator.reset()
    
    # Evaluate on validation set
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            
            # Get predictions
            preds = outputs.max(dim=1)[1]
            
            # Update metrics (using the model's existing evaluator)
            model.test_evaluator.update(
                targets.cpu().numpy(),
                preds.cpu().numpy()
            )
    
    # Print validation metrics
    val_metrics = model.test_evaluator.get_results()
    print("\n===== Validation Metrics =====")
    print(f"Mean IoU: {val_metrics['Mean IoU']:.4f}")
    print(f"Overall Acc: {val_metrics['Overall Acc']:.4f}")
    print(f"Mean Acc: {val_metrics['Mean Acc']:.4f}")

if __name__ == "__main__":
    main() 