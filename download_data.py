from datasets import load_dataset
import logging
import config

def load_data(dataset_id: str = config.dataset_id):
    try:
        datasets = load_dataset(dataset_id)
        datasets.save_to_disk(config.dataset_dir)
        logging.info("Dataset downloaded successfully")

    except Exception as e:
        logging.ERROR(f"dataset downloading error {e}")


load_data()
