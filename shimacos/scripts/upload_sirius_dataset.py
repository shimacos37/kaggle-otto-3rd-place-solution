import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    model_name = "weighted_order_cart"
    df = pl.read_parquet(f"./output/sirius/oof_{model_name}.parquet")
    df.to_pandas().to_gbq(
        f"otto_valid_result.{model_name}",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "aid", "type": "INT64"},
            {"name": "pred_clicks", "type": "FLOAT64"},
            {"name": "pred_carts", "type": "FLOAT64"},
            {"name": "pred_orders", "type": "FLOAT64"},
        ],
        if_exists="replace",
    )
    df = pl.read_parquet(f"./output/sirius/pred_{model_name}.parquet")
    df.to_pandas().to_gbq(
        f"otto_test_result.{model_name}",
        project_id=config.env.project_id,
        table_schema=[
            {"name": "session", "type": "INT64"},
            {"name": "aid", "type": "INT64"},
            {"name": "pred_clicks", "type": "FLOAT64"},
            {"name": "pred_carts", "type": "FLOAT64"},
            {"name": "pred_orders", "type": "FLOAT64"},
        ],
        if_exists="replace",
    )


if __name__ == "__main__":
    main()
