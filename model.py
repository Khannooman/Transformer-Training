from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import config
import logging

def load_model(model: str = config.model_name):
    """load the model from huggingface"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model)
        hidden_size = config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, config.number_of_class)
        )
        return model
    
    except Exception as e:
        logging.ERROR(f"model loading error {e}")


        