from typing import Dict, List

import hydra
import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from cuml.neighbors import NearestNeighbors
except:
    from sklearn.neighbors import NearestNeighbors


def upload_df(
    df: pd.DataFrame, project_id: str, table_name: str, schema: List[Dict[str, str]]
):
    for index, df_chunk in enumerate(tqdm(np.array_split(df, 10))):
        if index == 0:
            if_exists = "replace"
        else:
            if_exists = "append"
        df_chunk.to_gbq(
            f"otto_preprocess_003.{table_name}",
            project_id=project_id,
            table_schema=schema,
            if_exists=if_exists,
        )


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    for dataset_name in ["train_valid", "train_test"]:
        df = pd.read_gbq(
            f"""
            select distinct
                session,
                aid,
            from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            """,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm",
        )
        session_ids = pd.read_gbq(
            f"""
            select distinct
                session,
            from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            where
            split != 'train'
            """,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm",
        )["session"].to_numpy()
        aids = pd.read_gbq(
            f"""
            select distinct
                aid,
            from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            """,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm",
        )["aid"].to_numpy()

        print(df.shape)
        df["user_label"], user_idx = pd.factorize(df["session"])
        df["item_label"], item_idx = pd.factorize(df["aid"])
        sparse_item_user = sparse.csr_matrix(
            (np.ones(len(df)), (df["user_label"], df["item_label"]))
        )

        epoch, emb_size = 6000, 64
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=emb_size, regularization=0.001, iterations=epoch, random_state=777
        )
        model.fit(sparse_item_user)
        pd.to_pickle(model, f"./input/bpr_{dataset_name}_003.pkl")
        model = pd.read_pickle(f"./input/bpr_{dataset_name}_003.pkl")
        u2emb = dict(zip(user_idx, model.user_factors.to_numpy()))
        i2emb = dict(zip(item_idx, model.item_factors.to_numpy()))
        user_df = pd.DataFrame({"session": session_ids})
        user_vectors = [u2emb.get(id_, np.zeros(emb_size + 1)) for id_ in session_ids]
        user_df["session_vector"] = user_vectors
        aid_df = pd.DataFrame({"aid": aids})
        aid_vectors = [i2emb.get(id_, np.zeros(emb_size + 1)) for id_ in aids]
        aid_df["aid_vector"] = aid_vectors

        knn = NearestNeighbors(n_neighbors=50, metric="cosine")
        knn.fit(np.stack(aid_vectors))
        distances, indices = knn.kneighbors(np.stack(user_vectors))
        neighbor_aids = aids[indices]
        bpr_df = pd.DataFrame({"session": session_ids})
        bpr_df["neighbor_aids"] = neighbor_aids.tolist()
        cosine_similarity = 1 - distances
        bpr_df["cosine_similarity"] = cosine_similarity.tolist()
        upload_df(
            bpr_df,
            config.env.project_id,
            f"bpr_cosine_similarity_{dataset_name}",
            schema=[
                {"name": "session", "type": "INT64"},
                {"name": "neighbor_aids", "type": "INT64", "mode": "REPEATED"},
                {"name": "cosine_similarity", "type": "FLOAT64", "mode": "REPEATED"},
            ],
        )
        upload_df(
            user_df,
            config.env.project_id,
            f"bpr_user_vectors_{dataset_name}",
            schema=[
                {"name": "session", "type": "INT64"},
                {"name": "session_vector", "type": "FLOAT64", "mode": "REPEATED"},
            ],
        )
        upload_df(
            aid_df,
            config.env.project_id,
            f"bpr_aid_vectors_{dataset_name}",
            schema=[
                {"name": "aid", "type": "INT64"},
                {"name": "aid_vector", "type": "FLOAT64", "mode": "REPEATED"},
            ],
        )


if __name__ == "__main__":
    main()
