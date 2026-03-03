from huggingface_hub import HfApi, login, upload_folder
import os

repo_id = "Lullooo/BTC-volatility-forecasting-model"

login(token=os.environ["HF_TOKEN"])

upload_folder(
    folder_path="models",
    repo_id=repo_id,
    repo_type="model",
)