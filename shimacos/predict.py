import logging
import os
import pickle
import random
from glob import glob
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import catboost as cat
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import storage
from omegaconf import DictConfig
from tqdm import tqdm, trange

from src.factories.tree_factory import LGBMModel
from src.utils import read_gbq_allow_large_results

try:
    from cuml import ForestInference

    USE_FOREST_INFERENCE = True
except:
    USE_FOREST_INFERENCE = False


plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="yamls", config_name="tree.yaml")
def main(config: DictConfig) -> None:
    os.chdir(config.workdir)
    models_dict = {}
    types = ["click", "cart_order"]
    for type in types:
        with open(f"{config.store.model_path}/booster_{type}_label.pkl", "rb") as f:
            bst = pickle.load(f)
            models_dict[type] = bst

    query = f"""
        select * from `{config.env.project_id}.otto_{config.feature.version}.train_test_dataset`
        join `{config.env.project_id}.otto_preprocess.test_session_ids_{{i}}` using(session)
        where aid is not null
    """
    results = []
    for mod in trange(0, 10):
        save_path = f"{config.store.result_path}/test_{mod}"
        if all([os.path.exists(f"{save_path}_{type}.pkl") for type in types]):
            pred_dfs = [pd.read_pickle(f"{save_path}_{type}.pkl") for type in types]
            results.extend(pred_dfs)
        else:
            _query = query.format(i=mod)
            _df = read_gbq_allow_large_results(
                _query,
                project_id=config.env.project_id,
                dataset_id=f"otto_{config.feature.version}",
                table_id="tmp_train_test_dataset",
            )
            feature_cols = [
                col for col in _df.columns if col not in config.feature.remove_cols
            ]
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(_df[col]):
                    _df[col] = _df[col].astype(float).fillna(np.nan)
            for type in types:
                if os.path.exists(f"{save_path}_{type}.pkl"):
                    pred_df = pd.read_pickle(f"{save_path}_{type}.pkl")
                else:
                    if "xgb" in config.store.model_name:
                        preds = [
                            models_dict[type][n_fold].predict(
                                xgb.DMatrix(_df[feature_cols])
                            )
                            for n_fold in range(config.lgbm.n_fold)
                        ]
                    elif "cat" in config.store.model_name:
                        preds = [
                            models_dict[type][n_fold].predict(
                                cat.Pool(_df[feature_cols])
                            )
                            for n_fold in range(config.catboost.n_fold)
                        ]
                    else:
                        preds = [
                            models_dict[type][n_fold].predict(_df[feature_cols])
                            for n_fold in range(config.lgbm.n_fold)
                        ]
                    _df[f"{type}_pred"] = np.mean(preds, axis=0)
                    _df = _df.sort_values(["session", f"{type}_pred"], ascending=False)
                    tmp = _df.groupby("session").head(50)
                    if type == "click":
                        pred_df1 = tmp.groupby("session")["aid"].apply(
                            lambda ids: " ".join([str(int(id_)) for id_ in ids])
                        )
                        pred_df1.name = "labels"
                        pred_df1.index.name = "session_type"
                        pred_df1.index = pred_df1.index.astype("str") + f"_{type}s"
                        pred_df1 = pred_df1.reset_index()
                        pred_df2 = tmp.groupby("session")[f"{type}_pred"].apply(list)
                        pred_df2.name = "scores"
                        pred_df2.index.name = "session_type"
                        pred_df2.index = pred_df2.index.astype("str") + f"_{type}s"
                        pred_df2 = pred_df2.reset_index()
                        pred_df = pred_df1.merge(pred_df2, on="session_type")
                        pred_df.to_pickle(f"{save_path}_{type}.pkl")
                    else:
                        pred_df1 = tmp.groupby("session")["aid"].apply(
                            lambda ids: " ".join([str(int(id_)) for id_ in ids])
                        )
                        pred_df1.name = "labels"
                        pred_df1.index.name = "session_type"
                        pred_df1.index = pred_df1.index.astype("str") + "_carts"
                        pred_df1 = pred_df1.reset_index()
                        pred_df2 = tmp.groupby("session")["cart_order_pred"].apply(list)
                        pred_df2.name = "scores"
                        pred_df2.index.name = "session_type"
                        pred_df2.index = pred_df2.index.astype("str") + "_carts"
                        pred_df2 = pred_df2.reset_index()
                        pred_df = pred_df1.merge(pred_df2, on="session_type")
                        _pred_df = pred_df.copy()
                        _pred_df["session_type"] = _pred_df["session_type"].str.replace(
                            "cart", "order"
                        )
                        pred_df = pd.concat([pred_df, _pred_df])
                        pred_df.to_pickle(f"{save_path}_{type}.pkl")
                results.append(pred_df)
    results = pd.concat(results)
    results[["session_type", "labels"]].to_csv(
        f"{config.store.save_path}/sub.csv", index=False
    )
    results.to_pickle(f"{config.store.result_path}/test.pkl")
    for index, df_chunk in enumerate(tqdm(np.array_split(results, 10))):
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


if __name__ == "__main__":
    main()
