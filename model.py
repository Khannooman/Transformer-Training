from transformers import AutoModelForSequenceClassification
import config
import logging

def load_model(model: str = config.model_name):
    """load the model from huggingface"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model)
        return model
    
    except Exception as e:
        logging.ERROR(f"model loading error {e}")


        