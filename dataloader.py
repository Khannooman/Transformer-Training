from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

import torch    
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from  datasets import load_from_disk
from datasets import Dataset, load_from_disk
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 
class DataModule(Dataset):
    """Handles data loading, tokenization, and DatLoader Creation"""

    def __init__(self, config, part: str):
        """
        Initialize DataModule with configration
        """
        self.config = config
        self._data = self.load_dataset(f"{self.config.dataset_dir}/{part}")
        self.texts = self._data["text"]
        self.tokenizer = None
        self._load_tokenizer()

    def load_dataset(self, filepath: Path) -> Dataset:
        """Load dataset from disk with error handling"""
        try:
            filepath = Path(filepath)
            data = load_from_disk(str(filepath))
            logger.info(f"Successfully loaded dataset from {filepath}")
            return data
        except Exception as e:
            logging.error(f"Failed to load from {filepath}: {str(e)}")
            raise


    def _load_tokenizer(self):
        """load the tokenizer from huggingface"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.info(f"Successfully loaded tokenizer for {self.config.model_name}")
        except Exception as e:
            logging.error(f"model loading error {str(e)}")
            raise


    def tokenize_data(self,  text: str) -> Dict[str, torch.Tensor]:
        """tokenize text data in batches"""
        if self.tokenizer is None:
          self._load_tokenizer()
            
        try:
            encodings = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
                )
            return {
                'input_ids': encodings['input_ids'].squeeze(0),  # Remove batch dimension
                'attention_mask': encodings['attention_mask'].squeeze(0)
                }
        
        except Exception as e:
            logging.error(f"Tokenization failed {str(e)}")
            raise

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        try:
            item = self._data[idx]
            text = item["text"]
            label = item["label"]
            encoding = self.tokenize_data(text)
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"Error getting item at index {idx}: {str(e)}")
            raise


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
  """
  create train, validation, and test DataLoaders 
  """
  loaders = []

  for split in ['train', 'validation', 'test']:
      try:
          dataset = DataModule(config=config, part=split)

          ### Create loader
          loader = DataLoader(
              dataset,
              batch_size = config.batch_size,
              shuffle = split == 'train'
          )
          loaders.append(loader)
          logger.info("Succedully created dataloader")
  
      except Exception as e:
          logger.error(f"Error creating data loader {split}: {str(e)}")
          raise
  return tuple(loaders)

 
