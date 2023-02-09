import sys
from glob import glob

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(".")
from queries.bq import BQClient


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    bq = BQClient(config.env.project_id)
    bq.create_dataset("otto_preprocess_003")
    paths = glob("./input/train/train_parquet/*")
    train_df = pl.concat([pl.read_parquet(path) for path in paths])
    paths = glob("./input/train/test_parquet/*")
    test_df = pl.concat([pl.read_parquet(path) for path in paths])
    train_df = train_df.with_column(pl.lit("train").alias("split"))
    test_df = test_df.with_column(pl.lit("valid").alias("split"))
    df = pl.concat([train_df, test_df]).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for index, df_chunk in enumerate(tqdm(np.array_split(df, 10))):
        if index == 0:
            if_exists = "replace"
        else:
            if_exists = "append"
        df_chunk.to_gbq(
            "otto_preprocess_003.train_valid",
            project_id=config.env.project_id,
            table_schema=[
                {"name": "session", "type": "INT64"},
                {"name": "aid", "type": "INT64"},
                {"name": "ts", "type": "TIMESTAMP"},
                {"name": "type", "type": "STRING"},
                {"name": "split", "type": "STRING"},
            ],
            if_exists=if_exists,
        )
    # ======
    # test
    # ======
    train_df = pl.read_parquet("./input/test/train.parquet")
    test_df = pl.read_parquet("./input/test/test.parquet")
    train_df = train_df.with_column(pl.lit("train").alias("split"))
    test_df = test_df.with_column(pl.lit("test").alias("split"))
    df = pl.concat([train_df, test_df]).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], unit="s")

    df["type"] = df["type"].map({0: "clicks", 1: "carts", 2: "orders"})
    for index, df_chunk in enumerate(tqdm(np.array_split(df, 10))):
        if index == 0:
            if_exists = "replace"
        else:
            if_exists = "append"
        df_chunk.to_gbq(
            "otto_preprocess_003.train_test",
            project_id=config.env.project_id,
            table_schema=[
                {"name": "session", "type": "INT64"},
                {"name": "aid", "type": "INT64"},
                {"name": "ts", "type": "TIMESTAMP"},
                {"name": "type", "type": "STRING"},
                {"name": "split", "type": "STRING"},
            ],
            if_exists=if_exists,
        )
    gt_df = pl.read_parquet("./input/train/test_labels.parquet")
    click_df = (
        gt_df.filter(pl.col("type") == "clicks")
        .rename({"ground_truth": "click_labels"})
        .drop("type")
    )
    click_df = click_df.with_columns(
        click_df["click_labels"].apply(lambda x: x[0]).alias("click_label")
    )
    cart_df = (
        gt_df.filter(pl.col("type") == "carts")
        .rename({"ground_truth": "cart_label"})
        .drop("type")
    )
    order_df = (
        gt_df.filter(pl.col("type") == "orders")
        .rename({"ground_truth": "order_label"})
        .drop("type")
    )
    gt_df = (
        click_df.join(cart_df, on="session", how="outer")
        .join(order_df, on="session", how="outer")
        .to_pandas()
    )
    gt_df.to_gbq(
        "otto_preprocess_003.ground_truth",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "click_labels", "type": "INT64", "mode": "REPEATED"},
            {"name": "click_label", "type": "INT64"},
            {"name": "cart_label", "type": "INT64", "mode": "REPEATED"},
            {"name": "order_label", "type": "INT64", "mode": "REPEATED"},
        ],
        if_exists="replace",
    )
    gt_df.to_pickle("./input/ground_truth_003.pkl")


if __name__ == "__main__":
    main()
