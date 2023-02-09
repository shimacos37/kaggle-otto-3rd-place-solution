import logging

import numpy as np
import pandas as pd

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
    df["click_hit_num"] = df.apply(
        lambda row: click_hit_num(row["click_label"], row["click_pred"][:k]), axis=1
    )
    gt = df["click_label"].notnull().sum()
    click_recall = df["click_hit_num"].sum() / gt
    logger.info(f"click: {click_recall}")
    df["cart_hit_num"] = df.apply(
        lambda row: hit_num(row["cart_label"], row["cart_pred"][:k]), axis=1
    )
    gt = (np.minimum(df["cart_label"].apply(len), 20)).sum()
    cart_recall = df["cart_hit_num"].sum() / gt
    logger.info(f"cart: {cart_recall}")
    df["order_hit_num"] = df.apply(
        lambda row: hit_num(row["order_label"], row["order_pred"][:k]), axis=1
    )
    gt = (np.minimum(df["order_label"].apply(len), 20)).sum()
    order_recall = df["order_hit_num"].sum() / gt
    logger.info(f"order: {order_recall}")
    return click_recall * 0.1 + cart_recall * 0.3 + order_recall * 0.6
