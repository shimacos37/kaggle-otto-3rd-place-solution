import os
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

try:
    from cuml.neighbors import NearestNeighbors
except:
    from sklearn.neighbors import NearestNeighbors


tqdm.pandas()


import time

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for dataset_name in ["train_valid", "train_test"]:
        df = pd.read_gbq(
            f"""
            with base as (
                select
                  session,
                  aid,
                  ts,
                from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            ), encode as (
                select
                  aid,
                  row_number() over(order by aid) - 1 as new_aid,
                from (select distinct aid from base)
            )
            select *
            from (
                select distinct
                  lag(new_aid) over(partition by session order by ts) as lag_aid,
                  new_aid as aid,
                from base
                left join encode using(aid)
            )
            where lag_aid is not null
            """,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm",
        )
        aid_map = (
            pd.read_gbq(
                f"""
            with base as (
                select
                  session,
                  aid,
                  ts,
                from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            ), encode as (
                select
                  aid,
                  row_number() over(order by aid) - 1 as new_aid,
                from (select distinct aid from base)
            )
            select * from encode
            """,
                project_id=config.env.project_id,
                use_bqstorage_api=True,
                progress_bar_type="tqdm",
            )
            .set_index("new_aid")["aid"]
            .to_dict()
        )

        edges = df[["lag_aid", "aid"]].to_numpy().astype(int)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Node2Vec(
            torch.from_numpy(edges).T,
            embedding_dim=128,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
        ).to(device)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=os.cpu_count())
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        for epoch in range(1, 31):
            loss = train()
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
        with torch.no_grad():
            model.eval()
            vectors = (
                model(torch.arange(edges.max() + 1, device=device))
                .detach()
                .cpu()
                .numpy()
            )
        vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1)[:, None]
        aids = np.arange(df["aid"].max() + 1)
        aids = np.array([aid_map[aid] for aid in aids])
        vector_df = pd.DataFrame({"aid": aids})
        vector_df["vector"] = vectors.tolist()
        upload_df(
            vector_df,
            config.env.project_id,
            f"node2vec_128_all_vectors_{dataset_name}",
            schema=[
                {"name": "aid", "type": "INT64"},
                {"name": "vector", "type": "FLOAT64", "mode": "REPEATED"},
            ],
        )

        knn = NearestNeighbors(n_neighbors=50, metric="cosine")
        knn.fit(vectors)
        distances, indices = knn.kneighbors(vectors)
        neighbor_aids = aids[indices]
        n2v_df = pd.DataFrame({"aid": aids})
        n2v_df["neighbor_aids"] = neighbor_aids.tolist()
        cosine_similarity = 1 - distances
        n2v_df["cosine_similarity"] = cosine_similarity.tolist()
        upload_df(
            n2v_df,
            config.env.project_id,
            f"node2vec_128_all_cosine_similarity_{dataset_name}",
            schema=[
                {"name": "aid", "type": "INT64"},
                {"name": "neighbor_aids", "type": "INT64", "mode": "REPEATED"},
                {"name": "cosine_similarity", "type": "FLOAT64", "mode": "REPEATED"},
            ],
        )


if __name__ == "__main__":
    main()
