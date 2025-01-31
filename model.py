from transformers import AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import torch.nn as nn
import config
import logging

def load_model(
    model: str = config.model_name,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05
    ):
    # """Prepare model for k-bit-training with low rank adapter"""
    try:
      ### quantization configration
      # quantization_config = BitsAndBytesConfig(
      #         load_in_4bit=True,
      #         bnb_4bit_quant_type="nf4",
      #         bnb_4bit_use_double_quant=True,
      #         bnb_4bit_compute_dtype=torch.float16
      #     )
      
      ## load the model for k-bit training
      model = AutoModelForSequenceClassification.from_pretrained(
              model,
              # quantization_config = quantization_config,
              trust_remote_code=True  
              )
      # model.config.use_cache = False

      # ## prepare moddel for kbit training
      # model = prepare_model_for_kbit_training(model)

      # ### Lora Config
      # lora_config = LoraConfig(
      #     r=lora_r,
      #     lora_alpha=lora_alpha,
      #     lora_dropout=lora_dropout,
      #     bias="none",
      #     task_type="SEQ_CLS",
      #     target_modules=[
      #         "attention.self.query",
      #         "attention.self.key",
      #         "attention.self.value",
      #         "attention.output.dense",
      #         "intermediate.dense",
      #         "output.dense",
      #         "pooler.dense"
      #     ]
      # )

      # ### Get PEFT MODEL
      # model = get_peft_model(model, lora_config)

      ### Attach Linear layer
      hidden_size = config.hidden_size
      model.classifier = nn.Sequential(
          nn.Linear(model.config.hidden_size, hidden_size),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Linear(hidden_size, config.number_of_class)
      )
      return model

    except Exception as e:
        logging.error(f"model loading error {e}")



 