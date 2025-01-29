import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from typing import List, Dict, Optional
import wandb
import logging
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime
import json
import numpy as np   

import config
from model import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer(nn.Module):
    """Training the classifier"""

    def __init__(self, config):
        """Initialize the trainer with config"""
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device)
        self.model = load_model()
        logger.info(f"Using device: {self.device}")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Iniitalize the optimizer"""

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any (nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,

            }
        ]

        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    

    def _setup_schedular(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Initialize the learning rate schedular"""

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def save_checkpoint(self, epoch, optimizer, scheduler, best_metric, output_dir):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric
        }

        checkpoint_path = os.path.join(
            output_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )

        torch.save(checkpoint, checkpoint_path)

        config = {
            'epoch': epoch,
            'best_metrics': float(best_metric),
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }

        with open(os.path.join(output_dir, "training_config.json"), 'w') as config_file:
            json.dump(config, config_file)

    
    def load_checkpoint(self, checkpoint_path, optimizer=None, schedular=None):
        """Load checkpoints"""

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            self.model.load_state_dict(checkpoint['optimizer_state_dict'])

        if schedular and checkpoint['scheduler_state_dict']:
            self.model.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['best_metric']

    

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        optimizer = self._setup_optimizer()
        num_training_steps = len(train_loader) * self.config.epochs
        schedular = self._setup_schedular(optimizer, num_training_steps)

        if self.config.use_wandb:
            wandb.init(
                project="TextClassifier",

                config={
                    "learning_rate": self.config.learning_rate,
                    "architecture": self.config.model_name,
                    "dataset": self.config.dataset_id,
                    "epochs": self.config.epochs
                }
            )

        best_val_loss = float('inf')

        for epoch in range(self.config.epochs):
            train_loss = 0

            self.model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} / {self.config.epochs}")

            for batch in progress_bar:
                print(batch["input_ids"])
                batch = {
                        k: v for k, v in batch.items()
                }
                print(batch["input_ids"])

                del batch['label']

                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                optimizer.step()
                schedular.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            
                if self.config.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": schedular.get_last_lr()[0]
                    })

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Avg training loss: {avg_train_loss}")
        
            #evaluation

            if val_loader:
                val_loss, val_accurracy = self.evaluate(val_loader)
                logger.info(f"Validaton loss {val_loss} Validation Accuracy {val_accurracy}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch, 
                        optimizer, 
                        schedular, 
                        best_val_loss, 
                        self.config.output_dir
                    )
                
                if self.config.use_wandb:
                    wandb.log({
                        "val_loss": best_val_loss,
                        "val_accuracy": val_accurracy,
                        "epoch": epoch
                    })

    
    def evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on validate data"""

        self.model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                        k: v.clone().detach().to(self.device) if isinstance(v, torch.Tensor)
                        else torch.tensor(v, dtype=torch.long, device=self.device)
                        if isinstance(v, (list, np.ndarray)) and all(isinstance(i, int) for i in v)
                        else v  # Keep as-is if not convertible
                        for k, v in batch.items()
                }
                
                label = batch["label"]
                del batch["label"]
                outputs = self.model(**batch)
                loss = outputs.loss

                val_loss += loss.item()

                prediction = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (label == prediction).sum().item()
                total_predictions += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        return avg_val_loss, accuracy




