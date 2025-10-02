import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import the model
from network import modeling

# Import the Cityscapes dataset
from datasets.cityscapes import Cityscapes
import utils.ext_transforms as et
from metrics import StreamSegMetrics

# Constants
BATCH_SIZE = 16
NUM_EPOCHS = 2
CROP_SIZE = 505  # Default crop size, can be adjusted when calling train_model

class DeepLabModel(L.LightningModule):
    def __init__(self, num_classes=19, output_stride=8, learning_rate=0.01, pretrained_model_path=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.save_hyperparameters()
        
        # Initialize StreamSegMetrics for mIoU calculation instead of Evaluator
        self.train_evaluator = StreamSegMetrics(num_classes)
        self.val_evaluator = StreamSegMetrics(num_classes)
        self.test_evaluator = StreamSegMetrics(num_classes)
        
        # Load model with pretrained backbone
        self.model = modeling.deeplabv3plus_resnet101(
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=True
        )
        
        # Load pretrained weights (but ignore incompatible classifier weights)
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location="cpu")
                model_dict = self.model.state_dict()
                
                # Filter out keys related to the classifier since we changed it
                if "model_state" in checkpoint:
                    pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if "classifier" not in k}
                else:
                    pretrained_dict = {k: v for k, v in checkpoint.items() if "classifier" not in k}
                    
                # Update only the matching weights (backbone)
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print(f"Successfully loaded pretrained weights from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Will continue with randomly initialized weights")
        
        # Freeze the backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters: {num_trainable_params}")

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        # Reset evaluator at the beginning of each epoch
        self.train_evaluator.reset()
    
    def on_validation_epoch_start(self):
        # Reset evaluator at the beginning of each epoch
        self.val_evaluator.reset()
        
        # Track whether this is a sanity check
        if hasattr(self, 'trainer') and self.trainer.sanity_checking:
            self._in_sanity_check = True
        else:
            self._in_sanity_check = False
    
    def on_test_epoch_start(self):
        # Reset evaluator at the beginning of test epoch
        self.test_evaluator.reset()
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, labels, ignore_index=255)
        
        # Calculate accuracy excluding ignored indices
        mask = (labels != 255)
        if mask.sum() > 0:  # Check if there are valid pixels
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        # Get predictions for mIoU
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        # Update using the update method instead of add_batch
        self.train_evaluator.update(targets, preds)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        # Calculate mIoU for training data
        metrics = self.train_evaluator.get_results()
        train_miou = metrics["Mean IoU"]
        self.log("train_miou", train_miou, prog_bar=True)
        print(f"Train mIoU: {train_miou:.4f}")

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, labels, ignore_index=255)
        
        # Skip metrics calculation during sanity checks to avoid confusion
        if hasattr(self, '_in_sanity_check') and self._in_sanity_check:
            return loss
        
        # Calculate accuracy excluding ignored indices
        mask = (labels != 255)
        if mask.sum() > 0:  # Check if there are valid pixels
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        # Get predictions for mIoU
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        # Update using the update method instead of add_batch
        self.val_evaluator.update(targets, preds)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate mIoU for validation data
        metrics = self.val_evaluator.get_results()
        val_miou = metrics["Mean IoU"]
        # Log detailed metrics 
        self.log("val_miou", val_miou, prog_bar=True)
        self.log("val_overall_acc", metrics["Overall Acc"])
        self.log("val_mean_acc", metrics["Mean Acc"])
        
        # Skip printing metrics during sanity checks
        if self.trainer.sanity_checking:
            return
            
        print(f"Validation mIoU: {val_miou:.4f}")
        print(f"Validation Metrics: {self.val_evaluator.to_str(metrics)}")

    def test_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, labels, ignore_index=255)
        
        # Calculate accuracy excluding ignored indices
        mask = (labels != 255)
        if mask.sum() > 0:  # Check if there are valid pixels
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("test_acc", acc, on_epoch=True)
        
        # Get predictions for mIoU during testing
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        # Update the test evaluator
        self.test_evaluator.update(targets, preds)
        
        self.log("test_loss", loss)
        return loss
    
    def on_test_epoch_end(self):
        # Calculate and log test metrics
        # Debug the confusion matrix
        cm = self.test_evaluator.confusion_matrix
        print(f"Confusion matrix statistics:")
        print(f" - Shape: {cm.shape}")
        print(f" - Contains NaN: {np.isnan(cm).any()}")
        print(f" - Contains Inf: {np.isinf(cm).any()}")
        print(f" - Sum: {cm.sum()}")
        print(f" - Row sums: {np.sum(cm, axis=1)}")
        print(f" - Column sums: {np.sum(cm, axis=0)}")
        
        # Handle potential division by zero in IoU calculation
        metrics = self.test_evaluator.get_results()
        test_miou = metrics["Mean IoU"]
        
        # Safety check for NaN values
        if np.isnan(test_miou):
            print("WARNING: NaN values detected in Mean IoU. Adding safety handling...")
            # Recalculate with safety handling
            IoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-10)
            IoU = np.nan_to_num(IoU, nan=0.0)  # Replace NaN with 0
            test_miou = np.nanmean(IoU)  # Compute mean ignoring remaining NaN values
            print(f"Recalculated test_miou with safety handling: {test_miou:.4f}")
        
        self.log("test_miou", test_miou)
        self.log("test_overall_acc", metrics["Overall Acc"])
        self.log("test_mean_acc", metrics["Mean Acc"])
        
        print(f"Test mIoU: {test_miou:.4f}")
        print(f"Test Metrics: {self.test_evaluator.to_str(metrics)}")

    def configure_optimizers(self):
        # Only train the segmentation head
        trainable_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        return optim.SGD(trainable_params, lr=self.learning_rate)

# Custom callback to track loss metrics
class LossTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_mious = []
        self.val_mious = []
        
        # Track epochs to handle validation not running every epoch
        self.current_epoch = 0
        self.last_val_epoch = -1
        self.val_interval = 1  # Will be updated based on trainer settings
        
    def on_fit_start(self, trainer, pl_module):
        # Determine validation interval from trainer
        self.val_interval = trainer.check_val_every_n_epoch
        print(f"Validation running every {self.val_interval} epochs")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        train_miou = trainer.callback_metrics.get("train_miou")
        
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_acc is not None:
            self.train_accs.append(train_acc.item())
        if train_miou is not None:
            self.train_mious.append(train_miou.item())
        
        # If this is not a validation epoch, use previous validation values or None
        if self.current_epoch % self.val_interval != 0:
            # If we have previous validation values, repeat the last one
            if self.last_val_epoch >= 0:
                if self.val_losses:
                    self.val_losses.append(self.val_losses[-1])
                if self.val_accs:
                    self.val_accs.append(self.val_accs[-1])
                if self.val_mious:
                    self.val_mious.append(self.val_mious[-1])
            # Otherwise, use None
            else:
                self.val_losses.append(None)
                self.val_accs.append(None)
                self.val_mious.append(None)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Update last validation epoch
        self.last_val_epoch = self.current_epoch
        
        # Get the metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        val_miou = trainer.callback_metrics.get("val_miou")
        
        # Replace any previous placeholder for this epoch
        if val_loss is not None:
            # Replace the last placeholder or append
            if self.current_epoch < len(self.val_losses):
                self.val_losses[self.current_epoch] = val_loss.item()
            else:
                self.val_losses.append(val_loss.item())
        
        if val_acc is not None:
            if self.current_epoch < len(self.val_accs):
                self.val_accs[self.current_epoch] = val_acc.item()
            else:
                self.val_accs.append(val_acc.item())
                
        if val_miou is not None:
            if self.current_epoch < len(self.val_mious):
                self.val_mious[self.current_epoch] = val_miou.item()
            else:
                self.val_mious.append(val_miou.item())
    
    def plot_metrics(self, save_path="metrics_plot.png"):
        plt.figure(figsize=(18, 5))
        
        # Get the epoch indices (x-axis) - ensure all arrays have same length
        max_len = min(len(self.train_losses), len(self.val_losses))
        epochs = range(max_len)
        
        # Trim arrays to same length
        train_losses = self.train_losses[:max_len]
        val_losses = self.val_losses[:max_len]
        train_accs = self.train_accs[:max_len]
        val_accs = self.val_accs[:max_len]
        train_mious = self.train_mious[:max_len]
        val_mious = self.val_mious[:max_len]
        
        # Filter out None values for plotting
        val_losses = [v if v is not None else float('nan') for v in val_losses]
        val_accs = [v if v is not None else float('nan') for v in val_accs]
        val_mious = [v if v is not None else float('nan') for v in val_mious]
        
        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accs, label='Training Accuracy')
        plt.plot(epochs, val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot mIoU
        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_mious, label='Training mIoU')
        plt.plot(epochs, val_mious, label='Validation mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Training and Validation mIoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Metrics plot saved to {save_path}")

# Custom checkpoint callback to save model with best mIoU
class BestmIoUCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath=None, filename=None, monitor='val_miou', save_top_k=1, mode='max'):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            save_top_k=save_top_k,
            mode=mode,
            verbose=True
        )

def train_model(
    num_epochs=NUM_EPOCHS, 
    crop_size=CROP_SIZE,
    dataset_root='/nfs/shared/cityscapes',
    pretrained_model_path="/nfs/students/duk/data/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar",
    learning_rate=0.01,
    output_stride=8,
    batch_size=BATCH_SIZE,
    resume_from_checkpoint=None  # Add this parameter for resuming training
):
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    # Define transforms based on the crop size
    train_transforms = et.ExtCompose([
        et.ExtRandomCrop(size=(crop_size, crop_size)),  # Maintain aspect ratio for Cityscapes
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Setup Cityscapes dataset and data loaders
    try:
        train_dataset = Cityscapes(
            root=dataset_root,
            split='train',
            transform=train_transforms
        )
        
        val_dataset = Cityscapes(
            root=dataset_root,
            split='val',
            transform=val_transforms
        )
        
        print(f"Loaded train dataset with {len(train_dataset)} samples")
        print(f"Loaded validation dataset with {len(val_dataset)} samples")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        raise
    
    # Initialize model or load from checkpoint
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model = DeepLabModel.load_from_checkpoint(resume_from_checkpoint)
        print("Successfully loaded model from checkpoint")
    else:
        model = DeepLabModel(
            num_classes=19,
            output_stride=output_stride,
            learning_rate=learning_rate,
            pretrained_model_path=pretrained_model_path
        )
    
    # Create the loss tracker
    loss_tracker = LossTracker()
    
    # Add early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=8,
        verbose=True,
        mode='min'
    )
    
    # Setup checkpoint directory
    save_dir = os.path.join(os.getcwd(), "checkpoints_no_batchnorm")
    os.makedirs(save_dir, exist_ok=True)
    
    # Add checkpoint callback to save the best model by mIoU
    checkpoint_callback = BestmIoUCheckpoint(
        dirpath=save_dir,
        filename=f'deeplabv3_no_bn_crop_{crop_size}_best_miou',
        monitor='val_miou',
        mode='max'
    )
    
    # Setup trainer
    trainer = L.Trainer(
        accelerator="auto", 
        devices="auto", 
        strategy="auto", 
        max_epochs=num_epochs, 
        check_val_every_n_epoch=2,
        callbacks=[loss_tracker, early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0  # Disable sanity checks
    )
    
    # Train the model
    if resume_from_checkpoint:
        # Resume training from checkpoint - will continue from last epoch
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    else:
        # Start training from scratch
        trainer.fit(model, train_loader, val_loader)
    
    # Plot and save the metrics
    loss_tracker.plot_metrics(os.path.join(save_dir, f"deeplabv3_no_bn_crop_{crop_size}.png"))
    
    # Return the best path and model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return model, checkpoint_callback.best_model_path

def test_model(model, dataset_root='/nfs/shared/cityscapes', batch_size=BATCH_SIZE):
    """
    Test the model on the validation dataset and return metrics.
    
    Args:
        model: The trained DeepLabModel to be evaluated
        dataset_root: Path to the Cityscapes dataset
        batch_size: Batch size for testing
        
    Returns:
        result: A list containing a dictionary of test metrics including:
            - test_loss: Cross entropy loss on test set
            - test_acc: Pixel accuracy on test set
            - test_miou: Mean IoU on test set
            - test_overall_acc: Overall accuracy on test set
            - test_mean_acc: Mean class accuracy on test set
    """
    # Setup validation data loader for testing
    test_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = Cityscapes(
        root=dataset_root,
        split='val',  # Using validation set for testing
        transform=test_transforms
    )
    
    print(f"Loaded validation dataset for testing with {len(val_dataset)} samples")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Test the model on validation set
    trainer = L.Trainer()
    result = trainer.test(model, val_loader)
    
    # Print detailed metrics
    if result and len(result) > 0:
        print("Test Results on Validation Set:")
        print(f"  - mIoU: {result[0].get('test_miou', 'N/A'):.4f}")
        print(f"  - Overall Accuracy: {result[0].get('test_overall_acc', 'N/A'):.4f}")
        print(f"  - Mean Class Accuracy: {result[0].get('test_mean_acc', 'N/A'):.4f}")
    
    return result

if __name__ == "__main__":
    try:
        # Example of how to resume from a checkpoint:
        # CHECKPOINT_PATH = "checkpoints_no_batchnorm/deeplabv3_no_bn_crop_512_best_miou.ckpt"
        
        # To start a new training run:
        model, best_model_path = train_model(
            num_epochs=NUM_EPOCHS,
            crop_size=CROP_SIZE,
            # Override other params if needed
            # learning_rate=0.005,
            # batch_size=4,
            # resume_from_checkpoint=CHECKPOINT_PATH  # Uncomment to resume training
        )
        
        # Option 1: Test with the model from the last epoch (may not be the best)
        print("Testing with the final model (not necessarily the best model):")
        result_final = test_model(model)
        print("Final model test results:", result_final)
        
        # # Option 2: Load and test with the best model (by validation mIoU)
        # print("\nTesting with the best model (by validation mIoU):")
        # best_model = DeepLabModel.load_from_checkpoint(best_model_path)
        # result_best = test_model(best_model)
        # print("Best model test results:", result_best)
        
        print("Training and testing complete!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc() 