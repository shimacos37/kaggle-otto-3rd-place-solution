import logging
import os
import random
from glob import glob

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from google.cloud import storage
import wandb
from sklearn.preprocessing import LabelEncoder

from src.factories.tree_factory import LGBMModel
from src.utils import read_gbq, read_gbq_allow_large_results


plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def upload_directory(config: DictConfig) -> None:
    storage_client = storage.Client(config.env.project_id)
    bucket = storage_client.get_bucket(config.env.bucket_name)
    filenames = glob(os.path.join(config.store.save_path, "**"), recursive=True)
    for filename in filenames:
        if os.path.isdir(filename):
            continue
        destination_blob_name = os.path.join(
            config.store.gcs_path,
            filename.split(config.store.save_path)[-1][1:],
        )
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)


@hydra.main(config_path="yamls", config_name="tree.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    set_seed(config.seed)
    prepair_dir(config)

    df = read_gbq_allow_large_results(
        f"""
        SELECT * FROM `{config.env.project_id}.otto_{config.feature.version}.train_valid_sampled_dataset`
        ORDER BY session, aid
    """,
        project_id=config.env.project_id,
        dataset_id=f"otto_{config.feature.version}",
        table_id="tmp_train_valid_sampled_dataset",
    )
    df = df.sort_values(["session", "aid"], ignore_index=True)
    config.lgbm.feature_cols = [
        col for col in df.columns if col not in config.feature.remove_cols
    ]
    for col in config.lgbm.feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float).fillna(np.nan)
    le_dict = {}
    for col in config.lgbm.cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    label_cols = config.feature.label_cols
    for label_col in label_cols:
        config.lgbm.label_col = label_col
        config.lgbm.pred_col = f"{label_col}_pred"
        if label_col != "click_label":
            config.lgbm.params.learning_rate = 0.01
        model = LGBMModel(config.lgbm)
        df = model.cv(df)
        model.save_model(config.store.model_path)
        model.save_importance(config.store.result_path)
    pred_cols = [f"{col}_pred" for col in label_cols]
    df[["session", "aid"] + label_cols + pred_cols].to_pickle(
        f"{config.store.save_path}/valid.pickle"
    )
    # upload_directory(config)


if __name__ == "__main__":
    main()
