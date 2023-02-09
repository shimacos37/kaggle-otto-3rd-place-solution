import logging
import os
import pickle
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import polars as pl
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from wandb.lightgbm import wandb_callback as wandb_lgb_callback
from wandb.xgboost import wandb_callback as wandb_xgb_callback

import catboost as cat

plt.style.use("seaborn-whitegrid")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        importances = []
        for n_fold in range(self.config.n_fold):
            train_df = df.query("fold != @n_fold")
            valid_df = df.query("fold == @n_fold")
            bst = self.fit(train_df, valid_df)
            if self.config.params.objective == "multiclass":
                preds = bst.predict(valid_df[self.config.feature_cols])
                for i in range(self.config.params.num_class):
                    df.loc[valid_df.index, f"{self.config.label_col}_prob{i}"] = preds[
                        :, i
                    ]
            else:
                df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    valid_df[self.config.feature_cols]
                )
                if test_df is not None:
                    test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                        test_df[self.config.feature_cols]
                    )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance(importance_type="gain"))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        if test_df is not None:
            return df, test_df
        else:
            return df

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> lgb.Booster:
        params = dict(self.config.params)
        X_train = train_df[self.config.feature_cols]
        y_train = train_df[self.config.label_col]
        X_valid = valid_df[self.config.feature_cols]
        y_valid = valid_df[self.config.label_col]
        logger.info(
            f"{self.config.label_col} train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            # weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            # weight=train_df.query("fold==@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        if self.config.params.objective == "lambdarank":
            train_group = train_df["session"].value_counts(sort=False).to_list()
            valid_group = valid_df["session"].value_counts(sort=False).to_list()
            lgtrain.set_group(train_group)
            lgvalid.set_group(valid_group)
            params["ndcg_eval_at"] = [20, 30]
        evals_result = {}
        bst = lgb.train(
            params,
            lgtrain,
            num_boost_round=4000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=self.config.early_stopping_rounds,
            categorical_feature=self.config.cat_cols,
            verbose_eval=self.config.verbose_eval,
            evals_result=evals_result,
            # callbacks=[wandb_lgb_callback()]
            # feval=self._custom_metric,
        )
        logger.info(
            f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
        )
        return bst

    def second_fit(
        self,
        train_df: pd.DataFrame,
        num_iterations: int,
    ) -> lgb.Booster:
        params = dict(self.config.params)
        X_train = train_df[self.config.feature_cols]
        y_train = train_df[self.config.label_col]
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            # weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        if self.config.params.objective == "lambdarank":
            train_group = train_df["session"].value_counts(sort=False).to_list()
            lgtrain.set_group(train_group)
            params["ndcg_eval_at"] = [20, 30]
        bst = lgb.train(
            params,
            lgtrain,
            num_boost_round=num_iterations,
            valid_sets=[lgtrain],
            valid_names=["train"],
            categorical_feature=self.config.cat_cols,
            verbose_eval=self.config.verbose_eval,
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(
            f"{model_dir}/booster_{self.config.label_col + suffix}.pkl", "wb"
        ) as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class XGBModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, xgb.Booster] = {}

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(self.config.n_fold):
            train_df = df.query("fold != @n_fold")
            valid_df = df.query("fold == @n_fold")
            bst = self.fit(train_df, valid_df, pseudo_df)
            if self.config.params.objective == "multi:softmax":
                preds = bst.predict(xgb.DMatrix(valid_df[self.config.feature_cols]))
                test_preds = bst.predict(xgb.DMatrix(test_df[self.config.feature_cols]))
                for i in range(self.config.params.num_class):
                    df.loc[valid_df.index, f"{self.config.label_col}_prob{i}"] = preds[
                        :, i
                    ]
                    test_df[
                        f"{self.config.label_col}_prob{i}_fold{n_fold}"
                    ] = test_preds[:, i]
            else:
                df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    xgb.DMatrix(valid_df[self.config.feature_cols])
                )
                test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                    xgb.DMatrix(test_df[self.config.feature_cols])
                )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_score(importance_type="gain")
            importances.append(
                [
                    importance_dict[col] if col in importance_dict else 0
                    for col in self.config.feature_cols
                ]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        return df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        X_train = train_df[self.config.feature_cols]
        y_train = train_df[self.config.label_col]

        X_valid = valid_df[self.config.feature_cols]
        y_valid = valid_df[self.config.label_col]
        dtrain = xgb.DMatrix(
            X_train,
            label=np.array(y_train),
            feature_names=self.config.feature_cols,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            label=np.array(y_valid),
            feature_names=self.config.feature_cols,
        )
        if "rank" in self.config.params.objective:
            train_group = train_df["session"].value_counts(sort=False).to_list()
            valid_group = valid_df["session"].value_counts(sort=False).to_list()
            dtrain.set_group(train_group)
            dvalid.set_group(valid_group)
            self.config.params.validate_parameters = True

        bst = xgb.train(
            self.config.params,
            dtrain,
            num_boost_round=50000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,  # type: ignore
            # callbacks=[wandb_xgb_callback()],
            # feval=self.custom_metric,
        )
        return bst

    def second_fit(
        self,
        train_df: pd.DataFrame,
        num_iterations: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        X_train = train_df[self.config.feature_cols]
        y_train = train_df[self.config.label_col]

        dtrain = xgb.DMatrix(
            X_train,
            label=np.array(y_train),
            feature_names=self.config.feature_cols,
        )
        if "rank" in self.config.params.objective:
            train_group = train_df["session"].value_counts(sort=False).to_list()
            dtrain.set_group(train_group)
            self.config.params.validate_parameters = True

        bst = xgb.train(
            self.config.params,
            dtrain,
            num_boost_round=num_iterations,
            evals=[(dtrain, "train")],
            verbose_eval=100,  # type: ignore
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(
            f"{model_dir}/booster_{self.config.label_col + suffix}.pkl", "wb"
        ) as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class CatModel(object):
    """
    label_col毎にcatboost modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, cat.Booster] = {}

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        df: pd.DataFrame,
        test_df: Optional[pl.DataFrame] = None,
        pseudo_df: Optional[pl.DataFrame] = None,
    ) -> pd.DataFrame:
        importances = []
        preds = np.zeros(len(df))
        df = df.with_columns(pl.arange(0, len(df)).alias("index"))
        for n_fold in range(self.config.n_fold):
            train_df = df.filter(df["fold"] != n_fold)
            valid_df = df.filter(df["fold"] == n_fold)
            logger.info(
                f"{self.config.label_col}[fold {n_fold}] train shape: {train_df.shape}, valid shape: {valid_df.shape}"
            )
            bst, importance = self.fit(train_df, valid_df, pseudo_df)
            valid_pool = cat.Pool(
                valid_df[self.config.feature_cols].to_numpy(),
                cat_features=list(self.config.categorical_features_indices),
            )
            preds[valid_df["index"].to_numpy()] = bst.predict(valid_pool)
            if test_df is not None:
                test_preds = bst.predict(
                    cat.Pool(
                        test_df[self.config.feature_cols].to_numpy(),
                        cat_features=list(self.config.categorical_features_indices),
                    )
                )
                test_df = test_df.with_columns(
                    pl.Series(f"{self.config.pred_col}_fold{n_fold}", test_preds)
                )
            self.store_model(bst, n_fold)
            importances.append(importance)
        df = df.with_columns(pl.Series(self.config.pred_col, preds))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        if test_df is not None:
            return df, test_df
        else:
            return df

    def fit(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        pseudo_df: Optional[pl.DataFrame] = None,
    ) -> cat.CatBoost:

        X_train = train_df.select(self.config.feature_cols)
        y_train = train_df.select(self.config.label_col)

        X_valid = valid_df.select(self.config.feature_cols)
        y_valid = valid_df.select(self.config.label_col)
        dtrain = cat.Pool(
            X_train.to_numpy(),
            label=y_train.to_numpy(),
            feature_names=self.config.feature_cols,
            cat_features=list(self.config.categorical_features_indices),
        )
        dvalid = cat.Pool(
            X_valid.to_numpy(),
            label=y_valid.to_numpy(),
            feature_names=self.config.feature_cols,
            cat_features=list(self.config.categorical_features_indices),
        )
        if self.config.params.loss_function in [
            "YetiRank",
            "PairLogit",
            "PairLogitPairwise",
        ]:
            dtrain.set_group_id(train_df["session"].to_numpy())
            dvalid.set_group_id(valid_df["session"].to_numpy())

        bst = cat.train(
            pool=dtrain,
            params=dict(self.config.params),
            evals=dvalid,
            early_stopping_rounds=100,
            verbose_eval=100,
            # feval=self.custom_metric,
        )
        importance = bst.get_feature_importance(dtrain, type="FeatureImportance")
        return bst, importance

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(
            f"{model_dir}/booster_{self.config.label_col + suffix}.pkl", "wb"
        ) as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )
