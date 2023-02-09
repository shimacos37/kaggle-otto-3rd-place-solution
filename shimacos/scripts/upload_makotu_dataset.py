import hydra
from omegaconf import DictConfig
import polars as pl


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    model_names = ["makotu_v3", "makotu_v4"]
    for model_name in model_names:
        if model_name == "makotu_v3":
            types = ["cart", "order"]
        else:
            types = ["click", "cart", "order"]
        for type in types:
            if type == "click":
                df = (
                    pl.read_parquet(
                        f"./output/makotu/{type}_train_{model_name}.parquet"
                    )
                    .drop(["__index_level_0__"])
                    .rename({"pred": f"pred_{type}s"})
                )
            else:
                df = (
                    pl.read_parquet(
                        f"./output/makotu/{type}_train_{model_name}.parquet"
                    )
                    .drop(["__index_level_0__", "target"])
                    .rename({"pred": f"pred_{type}s"})
                )
            df.to_pandas().to_gbq(
                f"otto_valid_result.{type}_{model_name}",
                project_id=config.env.project_id,
                table_schema=[
                    {"name": "session", "type": "INT64"},
                    {"name": "aid", "type": "INT64"},
                    {"name": f"pred_{type}s", "type": "FLOAT64"},
                ],
                if_exists="replace",
            )
        for type in types:
            df = (
                pl.read_parquet(f"./output/makotu/{type}_test_{model_name}.parquet")
                .drop(["__index_level_0__"])
                .rename({"pred": f"pred_{type}s"})
            )
            df.to_pandas().to_gbq(
                f"otto_test_result.{type}_{model_name}",
                project_id=config.env.project_id,
                table_schema=[
                    {"name": "session", "type": "INT64"},
                    {"name": "aid", "type": "INT64"},
                    {"name": f"pred_{type}s", "type": "FLOAT64"},
                ],
                if_exists="replace",
            )

    # All click target (only v3)
    df = (
        pl.read_parquet("./output/makotu/click_train_makotu_v3_all_target.parquet")
        .drop(["__index_level_0__", "target"])
        .rename({"pred": "pred_all_clicks"})
    )
    df.to_pandas().to_gbq(
        f"otto_valid_result.all_click_{model_name}",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "aid", "type": "INT64"},
            {"name": "pred_all_clicks", "type": "FLOAT64"},
        ],
        if_exists="replace",
    )
    df = (
        pl.read_parquet("./output/makotu/click_test_makotu_v3_all_target.parquet")
        .drop(["__index_level_0__"])
        .rename({"pred": "pred_all_clicks"})
    )
    df.to_pandas().to_gbq(
        f"otto_test_result.all_click_{model_name}",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "aid", "type": "INT64"},
            {"name": "pred_all_clicks", "type": "FLOAT64"},
        ],
        if_exists="replace",
    )


if __name__ == "__main__":
    main()
