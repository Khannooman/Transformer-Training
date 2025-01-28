from pathlib import Path
from typing import Tuple, Optional

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
    def __init__(self, config):
        self.config = config
        self.tokenizer = Optional[PreTrainedTokenizer] = None


    def load_dataset(self, filepath: Path) -> Dataset:
        """Load dataset from disk with error handling"""
        try:
            data = load_from_disk(str(filepath))
            logger.info(f"Successfully loaded dataset from {filepath}")
            return data
        except Exception as e:
            logging.ERROR(f"data loading from disk error {e}")


    def load_tokenizer(self) -> PreTrainedTokenizer:
        """load the tokenizer from huggingface"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.info(f"Successfully loaded tokenizer for {self.config.model_name}")
            return self.tokenizer
        except Exception as e:
            logging.error(f"model loading error {str(e)}")
            raise
        


def tokenize_data(data):
    """tokenize text columns"""
    tokenizer = load_tokenizer()
    try:
        return tokenizer(data["text"], padding="max_length", truncation=True)
    except Exception as e:
        logging.ERROR(f"Tokenization error {e}")



def create_data_loader(parent_data_filepath: str):
    """create dataloader to load data in bacths"""

    train_data = load_data_from_disk("./data/train")
    train_data = train_data.map(tokenize_data, batched=True).remove_columns(["text"])

    test_data = load_data_from_disk("./data/test")
    test_data = test_data.map(tokenize_data, batched=True).remove_columns(["text"])
    val_data = load_data_from_disk("./data/val")
    val_data = val_data.map(tokenize_data, batched=True).remove_columns(["text"])

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True)
    
    val_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_loader, val_loader, test_loader
 
