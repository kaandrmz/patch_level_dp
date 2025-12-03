"""DP classification model using Lightning."""

import math
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from dp_accounting.pld.privacy_loss_distribution import identity

from ..data import get_classification_dataset_config
from ..models import get_model_architecture
from ..privacy import (
    setup_dp_model, setup_dp_optimizer, setup_batch_memory_manager,
    calc_noise, create_pld, calc_sampling_prob
)


class DPClassificationModel(L.LightningModule):
    """DP classification model supporting ResNet, SimpleConv, etc."""
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
        batch_sampling_prob: float,
        max_physical_batch_size: int,
        batch_size: int,
        clip_norm: float,
        num_queries: int,
        train_loader: Any,
        crop_size: int,
        privacy_patch_size: Tuple[int, int],
        padding: int,
        lr: float = 0.01,
        momentum: float = 0.9,
        standard_deviation: Optional[float] = None,
        baseline_privacy: bool = False,
        **model_kwargs
    ):
        """Initialize DP classification model.
        
        Args:
            model_name: Model architecture name (e.g., "resnet18", "simpleconv")
            dataset_name: Dataset name (e.g., "mnist", "dtd")
            epsilon: Privacy budget epsilon
            delta: Privacy budget delta
            sensitivity: Sensitivity parameter for DP
            batch_sampling_prob: Batch sampling probability
            max_physical_batch_size: Maximum physical batch size for memory management
            batch_size: Logical batch size
            clip_norm: Gradient clipping norm
            num_queries: Number of privacy queries
            train_loader: Training data loader
            crop_size: Crop size for training
            privacy_patch_size: Size of privacy-sensitive patches
            padding: Padding for images
            lr: Learning rate
            momentum: SGD momentum factor
            standard_deviation: Fixed noise level (if None, calculated from epsilon)
            baseline_privacy: If True, uses baseline privacy settings
            **model_kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.model_arch = get_model_architecture(model_name)
        self.dataset_config = get_classification_dataset_config(dataset_name)
        
        self.save_hyperparameters(ignore=["train_loader"])
        self.automatic_optimization = False
        
        self.model = self.model_arch.create_model(
            num_classes=self.dataset_config.num_classes,
            **model_kwargs
        )
        
        self._setup_batchnorm_for_dp()
        
        # Simple accuracy tracking for classification
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        self.test_correct = 0
        self.test_total = 0
        
        setup_dp_model(self.model)
        
        self._setup_privacy_parameters(
            epsilon, delta, sensitivity, batch_sampling_prob, 
            num_queries, crop_size, privacy_patch_size, padding, standard_deviation,
            baseline_privacy
        )
        
        self.hparams.max_physical_batch_size = max_physical_batch_size
        self.hparams.batch_size = batch_size
        self.hparams.delta = delta
        self.train_loader = train_loader
    
    def _setup_batchnorm_for_dp(self):
        """Remove BatchNorm layers for DP compatibility by replacing with Identity.""" # vs freezing bc training from scratch
        print("Removing all BatchNorm2d layers (replacing with Identity)...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                    attr_name = name
                setattr(parent, attr_name, nn.Identity())
    
    def _setup_privacy_parameters(
        self, epsilon, delta, sensitivity, batch_sampling_prob, 
        num_queries, crop_size, privacy_patch_size, padding, standard_deviation,
        baseline_privacy
    ):
        """Setup privacy parameters and calculations."""
        if standard_deviation is None:
            self.standard_deviation, sampling_prob = calc_noise(
                epsilon=epsilon,
                delta=delta,
                sensitivity=sensitivity,
                batch_sampling_prob=batch_sampling_prob,
                num_queries=num_queries,
                image_size=self.dataset_config.image_size,
                crop_size=crop_size,
                private_patch_size=privacy_patch_size,
                padding=padding,
                baseline_privacy=baseline_privacy,
            )
        else:
            self.standard_deviation = standard_deviation
            sampling_prob = calc_sampling_prob(
                image_size=self.dataset_config.image_size,
                crop_size=crop_size,
                private_patch_size=privacy_patch_size,
                padding=padding,
                batch_sampling_prob=batch_sampling_prob,
                baseline_privacy=baseline_privacy,
            )

        print(f"STANDARD_DEVIATION: {self.standard_deviation}")
        
        self.pld = create_pld(self.standard_deviation, sensitivity, sampling_prob)
        self.composed_pld = identity()
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def on_train_epoch_start(self):
        """Reset metrics at start of each training epoch."""
        print(f"Training epoch start, train_loader length: {len(self.train_loader)}")
        self.train_correct = 0
        self.train_total = 0
    
    def train_dataloader(self):
        """Setup training dataloader with batch memory manager."""
        bmm = setup_batch_memory_manager(
            data_loader=self.train_loader,
            max_physical_batch_size=self.hparams.max_physical_batch_size,
            optimizer=self.optimizer
        )
        bmm_iter = bmm.__enter__()
        return bmm_iter

    def training_step(self, batch, batch_idx):
        """Training step with DP-SGD."""
        x, labels = batch
        assert len(x) <= self.hparams.max_physical_batch_size
        
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        
        optimizer = self.optimizers()
        self.manual_backward(loss)  
        optimizer.step()            
        optimizer.zero_grad()       
        
        if (batch_idx+1) % (self.hparams.batch_size / self.hparams.max_physical_batch_size) == 0:
            self.composed_pld = self.composed_pld.compose(self.pld)
        
        # Classification accuracy
        preds = outputs.argmax(dim=1)
        self.train_correct += (preds == labels).sum().item()
        self.train_total += labels.size(0)
        
        return loss

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        train_acc = self.train_correct / max(self.train_total, 1)
        self.log("train_acc", train_acc)
        print(f"Train - Accuracy: {train_acc:.4f} ({self.train_correct}/{self.train_total})")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, labels = batch
        outputs = self.model(x)
        
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        
        # Classification accuracy
        preds = outputs.argmax(dim=1)
        self.val_correct += (preds == labels).sum().item()
        self.val_total += labels.size(0)
        
        return loss

    def on_validation_epoch_start(self):
        """Reset validation metrics."""
        self.val_correct = 0
        self.val_total = 0

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        val_acc = self.val_correct / max(self.val_total, 1)
        self.log("val_acc", val_acc)
        print(f"Validation - Accuracy: {val_acc:.4f} ({self.val_correct}/{self.val_total})")

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, labels = batch
        outputs = self.model(x)
        
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss)
        
        final_epsilon = self.composed_pld.get_epsilon_for_delta(self.hparams.delta)
        self.log("epsilon", final_epsilon, on_step=False, on_epoch=True)
        
        # Classification accuracy
        preds = outputs.argmax(dim=1)
        self.test_correct += (preds == labels).sum().item()
        self.test_total += labels.size(0)

        return loss

    def on_test_epoch_start(self):
        """Reset test metrics."""
        self.test_correct = 0
        self.test_total = 0

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        test_acc = self.test_correct / max(self.test_total, 1)
        self.log("test_acc", test_acc)
        print(f"Test - Accuracy: {test_acc:.4f} ({self.test_correct}/{self.test_total})")

    def configure_optimizers(self):
        """Configure optimizers for DP training."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(trainable_params, lr=self.hparams.lr, momentum=self.hparams.momentum)
        optimizer = optim.Adam(trainable_params, lr=self.hparams.lr)
        self.optimizer = setup_dp_optimizer(
            optimizer,
            # noise_multiplier=self.standard_deviation,
            noise_multiplier=0.0, # if you wanna make non dp
            max_grad_norm=self.hparams.clip_norm,
            expected_batch_size=self.hparams.batch_size,
        )
        return self.optimizer
