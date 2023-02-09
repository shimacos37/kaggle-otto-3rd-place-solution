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
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from src.factories.tree_factory import LGBMModel
from src.utils import read_gbq_allow_large_results, read_gbq
from src.metrics import compute_metric

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


@hydra.main(config_path="yamls", config_name="stacking.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    set_seed(config.seed)
    prepair_dir(config)
    label_cols = config.feature.label_cols
    gt_df = pd.read_pickle("./input/ground_truth_003.pkl")
    if not config.test:
        df = read_gbq_allow_large_results(
            f"""
            SELECT * FROM `{config.env.project_id}.otto_stacking_{config.feature.version}.train_dataset`
            ORDER BY session, aid
        """,
            project_id=config.env.project_id,
            dataset_id=f"otto_stacking_{config.feature.version}",
            table_id="tmp_train_valid_sampled_dataset",
        )
        df = df.sort_values(["session", "aid"], ignore_index=True)
        config.lgbm.feature_cols = [
            col for col in df.columns if col not in config.feature.remove_cols
        ]
        for col in config.lgbm.feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float).fillna(np.nan)
        gkf = GroupKFold(n_splits=5)
        for n_fold, (_, valid_idx) in enumerate(gkf.split(df, groups=df["session"])):
            df.loc[valid_idx, "fold"] = n_fold
        result_df = pd.DataFrame({"session": df["session"].unique()})
        gt_df = pd.read_pickle("./input/ground_truth_003.pkl")
        for label_col in label_cols:
            config.lgbm.label_col = label_col
            config.lgbm.pred_col = f"{label_col}_pred"
            config.lgbm.params.learning_rate = 0.01
            model = LGBMModel(config.lgbm)
            df = model.cv(df)
            # Valid
            model.save_model(config.store.model_path)
            model.save_importance(config.store.result_path)
            df = df.sort_values(["session", config.lgbm.pred_col], ascending=False)
            pred_df = df.groupby(["session"])["aid"].apply(list)
            pred_df.name = label_col.split("_")[0] + "_pred"
            result_df = result_df.merge(pred_df, on="session", how="left")
        result_df = result_df.merge(gt_df, on="session", how="left")
        result_df["click_label"] = result_df["click_label"].astype(float).fillna(np.nan)
        result_df.to_pickle(f"{config.store.save_path}/valid.pkl")
        logger.info(f"CV: {compute_metric(result_df)}")
    else:
        test_df = read_gbq_allow_large_results(
            f"""
            SELECT * FROM `{config.env.project_id}.otto_stacking_{config.feature.version}.test_dataset`
            ORDER BY session, aid
        """,
            project_id=config.env.project_id,
            dataset_id=f"otto_stacking_{config.feature.version}",
            table_id="tmp_train_test_sampled_dataset",
        )
        test_df = test_df.sort_values(["session", "aid"], ignore_index=True)
        config.lgbm.feature_cols = [
            col for col in test_df.columns if col not in config.feature.remove_cols
        ]
        for col in config.lgbm.feature_cols:
            if pd.api.types.is_numeric_dtype(test_df[col]):
                test_df[col] = test_df[col].astype(float).fillna(np.nan)
        sub = []
        for label_col in tqdm(label_cols):
            config.lgbm.label_col = label_col
            config.lgbm.pred_col = f"{label_col}_pred"
            bst_dict = pd.read_pickle(
                f"{config.store.model_path}/booster_{label_col}.pkl"
            )
            # Test
            for i in range(5):
                test_df[f"{config.lgbm.pred_col}_fold{i}"] = bst_dict[i].predict(
                    test_df[config.lgbm.feature_cols]
                )
            test_df[config.lgbm.pred_col] = test_df[
                [f"{config.lgbm.pred_col}_fold{i}" for i in range(5)]
            ].mean(1)
            test_df = test_df.sort_values(
                ["session", config.lgbm.pred_col], ascending=False
            )
            tmp = test_df.groupby("session").head(50)
            pred_df1 = tmp.groupby(["session"])["aid"].apply(
                lambda ids: " ".join([str(int(id_)) for id_ in ids])
            )
            pred_df1.name = "labels"
            pred_df1.index.name = "session_type"
            pred_df1.index = (
                pred_df1.index.astype("str") + f'_{label_col.split("_")[0]}s'
            )
            pred_df1 = pred_df1.reset_index()
            pred_df2 = tmp.groupby("session")[config.lgbm.pred_col].apply(list)
            pred_df2.name = "scores"
            pred_df2.index.name = "session_type"
            pred_df2.index = (
                pred_df2.index.astype("str") + f"_{label_col.split('_')[0]}s"
            )
            pred_df2 = pred_df2.reset_index()
            pred_df = pred_df1.merge(pred_df2, on="session_type")
            sub.append(pred_df)
        sub = pd.concat(sub)
        sub.to_pickle(f"{config.store.save_path}/test.pkl")
        sub[["session_type", "labels"]].to_csv(
            f"{config.store.save_path}/sub.csv", index=False
        )
        for index, df_chunk in enumerate(tqdm(np.array_split(sub, 10))):
            if index == 0:
                if_exists = "replace"
            else:
                if_exists = "append"
            df_chunk.to_gbq(
                f"otto_test_result.{config.store.model_name}",
                project_id=config.env.project_id,
                table_schema=[
                    {"name": "session_type", "type": "STRING"},
                    {"name": "labels", "type": "STRING"},
                    {"name": "scores", "type": "FLOAT64", "mode": "REPEATED"},
                ],
                if_exists=if_exists,
            )

    # upload_directory(config)


if __name__ == "__main__":
    main()
