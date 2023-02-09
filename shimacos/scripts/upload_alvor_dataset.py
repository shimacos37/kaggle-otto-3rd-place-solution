import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    for model_name in ["alvor_raw_predictions_596", "alvor_raw_predictions_599"]:
        if model_name == "alvor_raw_predictions_596":
            df = pl.read_parquet(
                "./output/alvor/alvor_oof_carts_orders.parquet"
            ).rename({"carts_pred": "pred_carts", "orders_pred": "pred_orders"})
            df2 = pl.read_parquet("./output/alvor/alvor_oof_clicks.parquet").rename(
                {"clicks_pred": "pred_clicks"}
            )
        else:
            df = pl.read_parquet(
                "./output/alvor/alvor_oof_carts_orders_v2.parquet"
            ).rename({"carts_pred": "pred_carts", "orders_pred": "pred_orders"})
            df2 = pl.read_parquet("./output/alvor/alvor_oof_clicks_v2.parquet").rename(
                {"clicks_pred": "pred_clicks"}
            )
        df = df.join(df2, on=["session", "aid"], how="outer")
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
        df = pl.read_parquet(f"./output/alvor/{model_name}.parquet").rename(
            {
                "clicks_pred": "pred_clicks",
                "carts_pred": "pred_carts",
                "orders_pred": "pred_orders",
            }
        )
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
