import itertools
import os
from itertools import chain

import hydra
import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from cuml.neighbors import NearestNeighbors
except:
    from sklearn.neighbors import NearestNeighbors


tqdm.pandas()


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print("Loss after epoch {}: {}".format(self.epoch, loss_now))
        self.epoch += 1


def remove_duplicates(lst):
    result = [key for key, _ in itertools.groupby(lst)]
    return result


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    for dataset_name in ["train_valid", "train_test"]:
        df = pd.read_gbq(
            f"""
            select
              session,
              array_agg(aid order by ts) as aids,
            from `{config.env.project_id}.otto_preprocess_003.{dataset_name}`
            group by 1
            having array_length(aids) > 1
            """,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm_notebook",
        )
        print(df.shape)
        aids = df["aids"].apply(list).to_list()
        aids = [remove_duplicates(a) for a in aids]
        model_name = (
            f"./input/word2vec/w2v_64_min_count5_window5_sg_003_{dataset_name}.model"
        )
        if os.path.exists(model_name):
            model = word2vec.Word2Vec.load(model_name)
        else:
            model = word2vec.Word2Vec(
                aids,
                vector_size=64,
                min_count=5,
                window=5,
                sg=1,
                sample=1e-4,
                ns_exponent=-0.5,
                epochs=100,
                workers=os.cpu_count(),
                compute_loss=True,
                callbacks=[callback()],
            )
            model.save(model_name)
        aids = np.unique(list(chain.from_iterable(aids)))
        vectors = np.array(
            [model.wv[aid] if aid in model.wv else np.zeros(64) for aid in tqdm(aids)]
        )
        vectors = vectors / np.linalg.norm(vectors, ord=2, axis=1)[:, None]
        vectors[np.isnan(vectors)] = 0
        w2v_df = pd.DataFrame({"aid": aids})
        w2v_df["vector"] = vectors.tolist()
        w2v_df.to_gbq(
            f"otto_preprocess_003.w2v_64_min_count5_window5_sg_vector_{dataset_name}",
            project_id=config.env.project_id,
            table_schema=[
                {"name": "aid", "type": "INT64"},
                {"name": "vector", "type": "FLOAT64", "mode": "REPEATED"},
            ],
            if_exists="replace",
        )
        knn = NearestNeighbors(n_neighbors=50, metric="cosine")
        knn.fit(vectors)
        distances, indices = knn.kneighbors(vectors)
        all_aids = aids[indices.sum(1) != -50]
        distances = distances[indices.sum(1) != -50]
        indices = indices[indices.sum(1) != -50]
        neighbor_aids = aids[indices]
        w2v_df = pd.DataFrame({"aid": all_aids})
        w2v_df["neighbor_aids"] = neighbor_aids.tolist()
        cosine_similarity = 1 - distances
        w2v_df["cosine_similarity"] = cosine_similarity.tolist()
        w2v_df.to_gbq(
            f"otto_preprocess_003.w2v_64_min_count5_window5_sg_cosine_similarity_{dataset_name}",
            project_id=config.env.project_id,
            table_schema=[
                {"name": "aid", "type": "INT64"},
                {"name": "neighbor_aids", "type": "INT64", "mode": "REPEATED"},
                {"name": "cosine_similarity", "type": "FLOAT64", "mode": "REPEATED"},
            ],
            if_exists="replace",
        )


if __name__ == "__main__":
    main()
