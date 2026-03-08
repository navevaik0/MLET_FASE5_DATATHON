import joblib
from pathlib import Path
import json
from datetime import datetime

def save_model(model, path: str):
    joblib.dump(model, path)

def save_dict(data: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_dict(path: str):
    with open(path) as f:
        return json.load(f)

def save_list(data: list, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_list(path: str):
    with open(path) as f:
        return json.load(f)
    
def update_metadata(metadata_path, model_name, version, metrics, features, champion=False):

    metadata_path = Path(metadata_path)

    metrics = {k: float(v) for k, v in metrics.items()}

    if metadata_path.exists():

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {"models": []}

    else:
        metadata = {"models": []}

    model_entry = {
        "model_name": model_name,
        "version": version,
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "metrics": metrics
    }

    metadata["models"].append(model_entry)

    if champion:
        metadata["champion_model"] = version

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)