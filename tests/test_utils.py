from src.utils import *
import joblib
from sklearn.linear_model import LogisticRegression


def test_save_model(tmp_path):

    model = {"model": "fake"}

    path = tmp_path / "model.joblib"

    save_model(model, path)

    loaded = joblib.load(path)

    assert loaded["model"] == "fake"

def test_save_and_load_dict(tmp_path):

    data = {"a": 1, "b": 2}

    path = tmp_path / "test_dict.json"

    save_dict(data, path)

    loaded = load_dict(path)

    assert loaded == data

def test_save_and_load_list(tmp_path):

    data = ["a", "b", "c"]

    path = tmp_path / "test_list.json"

    save_list(data, path)

    loaded = load_list(path)

    assert loaded == data

def test_update_metadata_create(tmp_path):

    path = tmp_path / "metadata.json"

    update_metadata(
        metadata_path=path,
        model_name="baseline",
        version="v1",
        metrics={"roc_auc": 0.75},
        features=["f1", "f2"],
        champion=True
    )

    with open(path) as f:
        metadata = json.load(f)

    assert metadata["models"][0]["model_name"] == "baseline"
    assert metadata["champion_model"] == "v1"

def test_update_metadata_append(tmp_path):

    path = tmp_path / "metadata.json"

    update_metadata(
        metadata_path=path,
        model_name="baseline",
        version="v1",
        metrics={"roc_auc": 0.70},
        features=["f1"],
    )

    update_metadata(
        metadata_path=path,
        model_name="random_forest",
        version="v2",
        metrics={"roc_auc": 0.85},
        features=["f1","f2"],
        champion=True
    )

    with open(path) as f:
        metadata = json.load(f)

    assert len(metadata["models"]) == 2
    assert metadata["champion_model"] == "v2"

def test_update_metadata_json_corrompido(tmp_path):

    path = tmp_path / "metadata.json"

    # cria JSON inválido
    with open(path, "w") as f:
        f.write("{invalid json}")

    update_metadata(
        metadata_path=path,
        model_name="model",
        version="v1",
        metrics={"roc_auc": 0.80},
        features=["f1"]
    )

    with open(path) as f:
        metadata = json.load(f)

    assert len(metadata["models"]) == 1