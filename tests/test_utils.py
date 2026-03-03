from src.utils import save_model
import tempfile
import os


def test_save_model_salva_arquivo():
    import joblib
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        save_model(model, path)
        assert os.path.exists(path)