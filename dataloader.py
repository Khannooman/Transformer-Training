from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

import torch    
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from  datasets import load_from_disk
from datasets import Dataset, load_from_disk
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataModule():
    """Handles data loading, tokenization, and DatLoader Creation"""

    def __init__(self, config):
        """
        Initialize DataModule with configration
        """
        self.config = config
        self.tokenizer = Optional[PreTrainedTokenizer] = None


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


    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """load the tokenizer from huggingface"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.info(f"Successfully loaded tokenizer for {self.config.model_name}")
            return self.tokenizer
        except Exception as e:
            logging.error(f"model loading error {str(e)}")
            raise


    def tokenize_data(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """tokenize text data in batches"""
        
        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()

        try:
            return self.tokenizer(
                batch["text"], 
                padding="max_length", 
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
                )
        
        except Exception as e:
            logging.error(f"Tokenization failed {str(e)}")
            raise

    def create_dataloaders(self, data_dir: Union[str, Path]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        create train, validation, and test DataLoaders 
        """
        data_dir = Path(data_dir)
        loaders = []

        for split in ['train', 'val', 'test']:
            try:
                dataset = self.load_dataset(data_dir / split)
                dataset.map(
                    self.tokenize_data,
                    batched=True,
                    remove_columns=["text"]
                )

                ### Create loader
                loader = DataLoader(
                    dataset,
                    batch_size = self.config.batch_size,
                    shuffle = split == 'train',
                    num_workers = self.config.num_workers
                )
                loaders.apppend(loader)
                logger.info("Succedully created dataloader")
            
            except Exception as e:
                logger.error(f"Error creating data loader {split}: {str(e)}")
                raise
        return tuple(loader)
    
