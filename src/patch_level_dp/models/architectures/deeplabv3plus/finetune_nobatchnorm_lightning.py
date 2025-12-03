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

from network import modeling
from datasets.cityscapes import Cityscapes
import utils.ext_transforms as et
from metrics import StreamSegMetrics

BATCH_SIZE = 16
NUM_EPOCHS = 2
CROP_SIZE = 505

class DeepLabModel(L.LightningModule):
    def __init__(self, num_classes=19, output_stride=8, learning_rate=0.01, pretrained_model_path=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.save_hyperparameters()
        
        self.train_evaluator = StreamSegMetrics(num_classes)
        self.val_evaluator = StreamSegMetrics(num_classes)
        self.test_evaluator = StreamSegMetrics(num_classes)
        
        self.model = modeling.deeplabv3plus_resnet101(
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=True
        )
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location="cpu")
                model_dict = self.model.state_dict()
                
                if "model_state" in checkpoint:
                    pretrained_dict = {k: v for k, v in checkpoint["model_state"].items() if "classifier" not in k}
                else:
                    pretrained_dict = {k: v for k, v in checkpoint.items() if "classifier" not in k}
                    
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print(f"Successfully loaded pretrained weights from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Will continue with randomly initialized weights")
        
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        trainable_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        num_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters: {num_trainable_params}")

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self.train_evaluator.reset()
    
    def on_validation_epoch_start(self):
        self.val_evaluator.reset()
        
        if hasattr(self, 'trainer') and self.trainer.sanity_checking:
            self._in_sanity_check = True
        else:
            self._in_sanity_check = False
    
    def on_test_epoch_start(self):
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
        
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        self.train_evaluator.update(targets, preds)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        metrics = self.train_evaluator.get_results()
        train_miou = metrics["Mean IoU"]
        self.log("train_miou", train_miou, prog_bar=True)
        print(f"Train mIoU: {train_miou:.4f}")

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, labels, ignore_index=255)
        
        if hasattr(self, '_in_sanity_check') and self._in_sanity_check:
            return loss
        
        mask = (labels != 255)
        if mask.sum() > 0:
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        self.val_evaluator.update(targets, preds)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        metrics = self.val_evaluator.get_results()
        val_miou = metrics["Mean IoU"]
        self.log("val_miou", val_miou, prog_bar=True)
        self.log("val_overall_acc", metrics["Overall Acc"])
        self.log("val_mean_acc", metrics["Mean Acc"])
        
        if self.trainer.sanity_checking:
            return
            
        print(f"Validation mIoU: {val_miou:.4f}")
        print(f"Validation Metrics: {self.val_evaluator.to_str(metrics)}")

    def test_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, labels, ignore_index=255)
        
        mask = (labels != 255)
        if mask.sum() > 0:
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("test_acc", acc, on_epoch=True)
        
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        self.test_evaluator.update(targets, preds)
        
        self.log("test_loss", loss)
        return loss
    
    def on_test_epoch_end(self):
        cm = self.test_evaluator.confusion_matrix
        print(f"Confusion matrix statistics:")
        print(f" - Shape: {cm.shape}")
        print(f" - Contains NaN: {np.isnan(cm).any()}")
        print(f" - Contains Inf: {np.isinf(cm).any()}")
        print(f" - Sum: {cm.sum()}")
        print(f" - Row sums: {np.sum(cm, axis=1)}")
        print(f" - Column sums: {np.sum(cm, axis=0)}")
        
        metrics = self.test_evaluator.get_results()
        test_miou = metrics["Mean IoU"]
        
        if np.isnan(test_miou):
            print("WARNING: NaN values detected in Mean IoU. Adding safety handling...")
            IoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-10)
            IoU = np.nan_to_num(IoU, nan=0.0)
            test_miou = np.nanmean(IoU)
            print(f"Recalculated test_miou with safety handling: {test_miou:.4f}")
        
        self.log("test_miou", test_miou)
        self.log("test_overall_acc", metrics["Overall Acc"])
        self.log("test_mean_acc", metrics["Mean Acc"])
        
        print(f"Test mIoU: {test_miou:.4f}")
        print(f"Test Metrics: {self.test_evaluator.to_str(metrics)}")

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        return optim.SGD(trainable_params, lr=self.learning_rate)

class LossTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_mious = []
        self.val_mious = []
        
        self.current_epoch = 0
        self.last_val_epoch = -1
        self.val_interval = 1
        
    def on_fit_start(self, trainer, pl_module):
        self.val_interval = trainer.check_val_every_n_epoch
        print(f"Validation running every {self.val_interval} epochs")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        train_miou = trainer.callback_metrics.get("train_miou")
        
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_acc is not None:
            self.train_accs.append(train_acc.item())
        if train_miou is not None:
            self.train_mious.append(train_miou.item())
        
        if self.current_epoch % self.val_interval != 0:
            if self.last_val_epoch >= 0:
                if self.val_losses:
                    self.val_losses.append(self.val_losses[-1])
                if self.val_accs:
                    self.val_accs.append(self.val_accs[-1])
                if self.val_mious:
                    self.val_mious.append(self.val_mious[-1])
            else:
                self.val_losses.append(None)
                self.val_accs.append(None)
                self.val_mious.append(None)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.last_val_epoch = self.current_epoch
        
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        val_miou = trainer.callback_metrics.get("val_miou")
        
        if val_loss is not None:
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
        
        max_len = min(len(self.train_losses), len(self.val_losses))
        epochs = range(max_len)
        
        train_losses = self.train_losses[:max_len]
        val_losses = self.val_losses[:max_len]
        train_accs = self.train_accs[:max_len]
        val_accs = self.val_accs[:max_len]
        train_mious = self.train_mious[:max_len]
        val_mious = self.val_mious[:max_len]
        
        val_losses = [v if v is not None else float('nan') for v in val_losses]
        val_accs = [v if v is not None else float('nan') for v in val_accs]
        val_mious = [v if v is not None else float('nan') for v in val_mious]
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accs, label='Training Accuracy')
        plt.plot(epochs, val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
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
    resume_from_checkpoint=None
):
    L.seed_everything(42)
    
    train_transforms = et.ExtCompose([
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
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
    
    loss_tracker = LossTracker()
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=8,
        verbose=True,
        mode='min'
    )
    
    save_dir = os.path.join(os.getcwd(), "checkpoints_no_batchnorm")
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_callback = BestmIoUCheckpoint(
        dirpath=save_dir,
        filename=f'deeplabv3_no_bn_crop_{crop_size}_best_miou',
        monitor='val_miou',
        mode='max'
    )
    
    trainer = L.Trainer(
        accelerator="auto", 
        devices="auto", 
        strategy="auto", 
        max_epochs=num_epochs, 
        check_val_every_n_epoch=2,
        callbacks=[loss_tracker, early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )
    
    if resume_from_checkpoint:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    loss_tracker.plot_metrics(os.path.join(save_dir, f"deeplabv3_no_bn_crop_{crop_size}.png"))
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return model, checkpoint_callback.best_model_path

def test_model(model, dataset_root='/nfs/shared/cityscapes', batch_size=BATCH_SIZE):
    """Test the model on the validation dataset and return metrics."""
    test_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = Cityscapes(
        root=dataset_root,
        split='val',
        transform=test_transforms
    )
    
    print(f"Loaded validation dataset for testing with {len(val_dataset)} samples")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    trainer = L.Trainer()
    result = trainer.test(model, val_loader)
    
    if result and len(result) > 0:
        print("Test Results on Validation Set:")
        print(f"  - mIoU: {result[0].get('test_miou', 'N/A'):.4f}")
        print(f"  - Overall Accuracy: {result[0].get('test_overall_acc', 'N/A'):.4f}")
        print(f"  - Mean Class Accuracy: {result[0].get('test_mean_acc', 'N/A'):.4f}")
    
    return result

if __name__ == "__main__":
    try:
        model, best_model_path = train_model(
            num_epochs=NUM_EPOCHS,
            crop_size=CROP_SIZE,
        )
        
        print("Testing with the final model (not necessarily the best model):")
        result_final = test_model(model)
        print("Final model test results:", result_final)
        
        print("Training and testing complete!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc() 