"""Generic DP segmentation model using Lightning."""

import math
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR, CosineAnnealingLR

import numpy as np
import lightning as L
from torchvision.models import ResNet101_Weights

from patch_level_dp.models.architectures.deeplabv3plus.metrics.stream_metrics import StreamSegMetrics
from dp_accounting.pld.privacy_loss_distribution import identity
from dp_accounting.pld import common

from ..data import get_dataset_config, DatasetConfig
from ..models import get_model_architecture, ModelArchitecture  
from ..privacy import (
    setup_dp_model, setup_dp_optimizer, setup_batch_memory_manager,
    calc_noise, create_pld, calc_sampling_prob
)


class DebugSGD(optim.SGD):
    """Debug version of SGD optimizer with additional logging."""
    
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(DebugSGD, self).__init__(params, lr=lr, momentum=momentum, 
                                     dampening=dampening, weight_decay=weight_decay, 
                                     nesterov=nesterov)
        self.step_count = 0
        self.lr = lr

    def step(self, closure=None):
        self.step_count += 1
        print(f"SGD step #{self.step_count} called with lr: {self.lr}")
        return super(DebugSGD, self).step(closure)


class DPSegmentationModel(L.LightningModule):
    """Generic DP segmentation model supporting multiple architectures and datasets."""
    
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
        standard_deviation: Optional[float] = None,
        baseline_privacy: bool = False,
        **model_kwargs
    ):
        """Initialize generic DP segmentation model.
        
        Args:
            model_name: Model architecture name (e.g., "deeplabv3plus", "pspnet")  
            dataset_name: Dataset name (e.g., "cityscapes", "a2d2")
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
            standard_deviation: Fixed noise level (if None, calculated from epsilon)
            baseline_privacy: If True, uses baseline privacy settings
            **model_kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.model_arch = get_model_architecture(model_name)
        self.dataset_config = get_dataset_config(dataset_name)
        
        self.save_hyperparameters(ignore=["train_loader"])
        self.automatic_optimization = False
        
        self.model = self.model_arch.create_model(
            num_classes=self.dataset_config.num_classes,
            **model_kwargs
        )
        
        self._setup_pretrained_backbone()
        self._setup_batchnorm_for_dp()
        
        self.train_evaluator = StreamSegMetrics(self.dataset_config.num_classes)
        self.val_evaluator = StreamSegMetrics(self.dataset_config.num_classes)
        self.test_evaluator = StreamSegMetrics(self.dataset_config.num_classes)
        
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
        
    def _setup_pretrained_backbone(self):
        """Setup pretrained backbone if supported by the model architecture."""
        if (self.model_arch.name == "deeplabv3plus" and 
            hasattr(self.model, 'backbone')):
            
            print("Loading ResNet101_Weights.IMAGENET1K_V2 for backbone...")
            try:
                weights = ResNet101_Weights.IMAGENET1K_V2
                resnet101_v2_state_dict = weights.get_state_dict(progress=True)
                
                self.model.backbone.load_state_dict(resnet101_v2_state_dict, strict=False)
                print("Successfully loaded ResNet101_Weights.IMAGENET1K_V2 into backbone.")
            except Exception as e:
                print(f"Error loading ResNet101_Weights.IMAGENET1K_V2: {e}")
                print("Proceeding without pretrained backbone weights.")
                
        elif self.model_arch.name == "pspnet":
            print("PSPNet handles pretrained backbone loading internally.")
    
    def _setup_batchnorm_for_dp(self):
        """Setup BatchNorm layers for DP compatibility."""
        print("Freezing all BatchNorm2d layers...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False 
    
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
        self.train_evaluator.reset()
    
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
        loss = self._compute_loss(outputs, labels)
        self.log("train_loss", loss)
        
        optimizer = self.optimizers()
        self.manual_backward(loss)  
        optimizer.step()            
        optimizer.zero_grad()       
        
        if (batch_idx+1) % (self.hparams.batch_size / self.hparams.max_physical_batch_size) == 0:
            self.composed_pld = self.composed_pld.compose(self.pld)
        
        if self.model_arch.requires_interpolation():
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.model_arch.get_loss_function_name() == "nll_loss":
            preds = torch.exp(outputs).detach().argmax(dim=1).cpu().numpy()
        else: # cross_entropy
            preds = outputs.detach().argmax(dim=1).cpu().numpy()
            
        targets = labels.detach().cpu().numpy()
        self.train_evaluator.update(targets, preds)
        
        mask = (labels != 255)
        if mask.sum() > 0:
            correct = ((outputs.argmax(1) == labels) * mask).sum().float()
            total = mask.sum().float()
            acc = correct / total
            self.log("train_acc", acc, on_step=False, on_epoch=True)
        else:
            self.log("No valid pixels in batch")
        
        return loss

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        score = self.train_evaluator.get_results()
        self.log("train_acc", score['Overall Acc'])
        self.log("train_miou", score['Mean IoU'])
        print(f"Train - Overall Acc: {score['Overall Acc']:.4f}, Mean IoU: {score['Mean IoU']:.4f}")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, labels = batch
        outputs = self.model(x)
        
        loss = self._compute_loss(outputs, labels)
        self.log("val_loss", loss)
        
        if self.model_arch.requires_interpolation():
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.model_arch.get_loss_function_name() == "nll_loss":
            preds = torch.exp(outputs).detach().argmax(dim=1).cpu().numpy()
        else: # cross_entropy
            preds = outputs.detach().argmax(dim=1).cpu().numpy()
            
        targets = labels.detach().cpu().numpy()
        self.val_evaluator.update(targets, preds)
        
        return loss

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        score = self.val_evaluator.get_results()
        self.log("val_acc", score['Overall Acc'])
        self.log("val_miou", score['Mean IoU'])
        print(f"Validation - Overall Acc: {score['Overall Acc']:.4f}, Mean IoU: {score['Mean IoU']:.4f}")
        self.val_evaluator.reset()

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, labels = batch
        outputs = self.model(x)
        
        loss = self._compute_loss(outputs, labels)
        self.log("test_loss", loss)
        
        final_epsilon = self.composed_pld.get_epsilon_for_delta(self.hparams.delta)
        self.log("epsilon", final_epsilon, on_step=False, on_epoch=True)

        if self.model_arch.requires_interpolation():
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.model_arch.get_loss_function_name() == "nll_loss":
            preds = torch.exp(outputs).detach().argmax(dim=1).cpu().numpy()
        else: # cross_entropy
            preds = outputs.detach().argmax(dim=1).cpu().numpy()
            
        targets = labels.detach().cpu().numpy()
        self.test_evaluator.update(targets, preds)

        return loss

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        score = self.test_evaluator.get_results()
        self.log("test_acc", score['Overall Acc'])
        self.log("test_miou", score['Mean IoU'])
        print(f"Test - Overall Acc: {score['Overall Acc']:.4f}, Mean IoU: {score['Mean IoU']:.4f}")
        self.test_evaluator.reset()

    def configure_optimizers(self):
        """Configure optimizers for DP training."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = DebugSGD(trainable_params, lr=self.hparams.lr)
        self.optimizer = setup_dp_optimizer(
            optimizer,
            noise_multiplier=self.standard_deviation,
            max_grad_norm=self.hparams.clip_norm,
            expected_batch_size=self.hparams.batch_size,
        )
        return self.optimizer
    
    def _compute_loss(self, outputs, labels):
        """Compute loss based on model architecture."""
        if self.model_arch.requires_interpolation():
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        loss_fn_name = self.model_arch.get_loss_function_name()
        
        if loss_fn_name == "cross_entropy":
            return F.cross_entropy(outputs, labels, ignore_index=255)
        elif loss_fn_name == "nll_loss":
            return F.nll_loss(outputs, labels, ignore_index=255)
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")