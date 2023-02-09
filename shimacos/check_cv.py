import logging
import os
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import GroupKFold
from src.utils import read_gbq, read_gbq_allow_large_results

tqdm.pandas()

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hit_num(gt, pred):
    if len(gt) == 0:
        return None
    gt = set(gt)
    pred = set(pred)
    return len(gt & pred)


def click_hit_num(gt, pred):
    if np.isnan(gt):
        return None
    return int(gt in pred)


def compute_metric(df: pd.DataFrame, k=20):
    df["click_hit_num"] = df.progress_apply(
        lambda row: click_hit_num(row["click_label"], row["click_pred"][:k]), axis=1
    )
    gt = df["click_label"].notnull().sum()
    click_recall = df["click_hit_num"].sum() / gt
    logger.info(f"click: {click_recall}")
    df["cart_hit_num"] = df.progress_apply(
        lambda row: hit_num(row["cart_label"], row["cart_pred"][:k]), axis=1
    )
    gt = (np.minimum(df["cart_label"].apply(len), k)).sum()
    cart_recall = df["cart_hit_num"].sum() / gt
    logger.info(f"cart: {cart_recall}")
    df["order_hit_num"] = df.progress_apply(
        lambda row: hit_num(row["order_label"], row["order_pred"][:k]), axis=1
    )
    gt = (np.minimum(df["order_label"].apply(len), k)).sum()
    order_recall = df["order_hit_num"].sum() / gt
    logger.info(f"order: {order_recall}")
    return click_recall * 0.1 + cart_recall * 0.3 + order_recall * 0.6


@hydra.main(config_path="yamls", config_name="tree.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    models_dict = {}
    types = ["click", "cart_order"]
    for type in types:
        with open(f"{config.store.model_path}/booster_{type}_label.pkl", "rb") as f:
            models_dict[type] = pickle.load(f)
    gt_df = pd.read_pickle("./input/ground_truth_003.pkl")
    query = f"""
      select * from `{config.env.project_id}.otto_{config.feature.version}.train_valid_dataset`
      join `{config.env.project_id}.otto_preprocess.valid_session_ids_{{i}}`
      using(session)
      where aid is not NULL
    """
    session2fold = (
        read_gbq(
            f"""
            select distinct session, fold 
            from `{config.env.project_id}.otto_{config.feature.version}.train_valid_sampled_dataset`
            """,
            project_id=config.env.project_id,
        )
        .set_index("session")["fold"]
        .to_dict(),
    )
    results = []
    for mod in tqdm([0, 1, 10, 11]):
        save_path = f"{config.store.result_path}/valid_{mod}"
        if os.path.exists(f"{save_path}.pkl"):
            pred_df = pd.read_pickle(f"{save_path}.pkl")
            results.append(pred_df)
        else:
            _query = query.format(i=mod)
            _df = read_gbq_allow_large_results(
                _query,
                project_id=config.env.project_id,
                dataset_id=f"otto_{config.feature.version}",
                table_id="tmp_train_valid_dataset",
            )
            _df["fold"] = _df["session"].map(session2fold)
            feature_cols = [
                col for col in _df.columns if col not in config.feature.remove_cols
            ]
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(_df[col]):
                    _df[col] = _df[col].astype(float).fillna(np.nan)
            valid_dfs = []
            for n_fold in range(config.lgbm.n_fold):
                valid_df = _df[_df["fold"] == n_fold]
                if len(valid_df) == 0:
                    continue
                result_df = pd.DataFrame({"session": valid_df["session"].unique()})
                for type in types:
                    if type == "click":
                        bst_dict = models_dict["click"]
                    else:
                        bst_dict = models_dict["cart_order"]
                    if "xgb" in config.store.model_name:
                        valid_df[f"{type}_pred"] = bst_dict[n_fold].predict(
                            xgb.DMatrix(valid_df[feature_cols])
                        )
                    elif "cat" in config.store.model_name:
                        valid_df[f"{type}_pred"] = bst_dict[n_fold].predict(
                            cat.Pool(valid_df[feature_cols])
                        )
                    else:
                        valid_df[f"{type}_pred"] = bst_dict[n_fold].predict(
                            valid_df[feature_cols]
                        )
                    valid_df = valid_df.sort_values(
                        ["session", f"{type}_pred"], ascending=False
                    )
                    tmp = valid_df.groupby("session").head(50)
                    pred_df = tmp.groupby("session")["aid"].apply(list)
                    pred_df.name = f"{type}_pred"
                    pred_df = pred_df.reset_index()
                    result_df = result_df.merge(pred_df, on="session", how="left")
                    pred_df = tmp.groupby("session")[f"{type}_pred"].apply(list)
                    pred_df.name = f"{type}_pred_score"
                    pred_df = pred_df.reset_index()
                    result_df = result_df.merge(pred_df, on="session", how="left")
                    if type == "cart_order":
                        result_df["cart_pred"] = result_df[f"{type}_pred"]
                        result_df["order_pred"] = result_df[f"{type}_pred"]
                        result_df["cart_pred_score"] = result_df[f"{type}_pred_score"]
                        result_df["order_pred_score"] = result_df[f"{type}_pred_score"]
                        result_df = result_df.drop(
                            [f"{type}_pred", f"{type}_pred_score"], axis=1
                        )
                valid_dfs.append(result_df)
            result_df = pd.concat(valid_dfs)
            result_df.to_pickle(f"{save_path}.pkl")
            tmp = result_df.merge(gt_df, on="session", how="left")
            tmp["click_label"] = tmp["click_label"].astype(float).fillna(np.nan)
            logger.info(f"CV ({mod}): {compute_metric(tmp)}")
            results.append(result_df)
    result_df = pd.concat(results)
    result_df = result_df.merge(gt_df, on="session", how="left")
    result_df["click_label"] = result_df["click_label"].astype(float).fillna(np.nan)
    logger.info(f"CV: {compute_metric(result_df)}")
    result_df.to_pickle(f"{config.store.save_path}/valid.pkl")
    result_df[
        [
            "session",
            "click_pred",
            "cart_pred",
            "order_pred",
            "click_pred_score",
            "cart_pred_score",
            "order_pred_score",
        ]
    ].to_gbq(
        f"otto_valid_result.{config.store.model_name}",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "click_pred", "type": "INT64", "mode": "REPEATED"},
            {"name": "cart_pred", "type": "INT64", "mode": "REPEATED"},
            {"name": "order_pred", "type": "INT64", "mode": "REPEATED"},
            {"name": "click_pred_score", "type": "FLOAT64", "mode": "REPEATED"},
            {"name": "cart_pred_score", "type": "FLOAT64", "mode": "REPEATED"},
            {"name": "order_pred_score", "type": "FLOAT64", "mode": "REPEATED"},
        ],
        if_exists="replace",
    )


if __name__ == "__main__":
    main()
