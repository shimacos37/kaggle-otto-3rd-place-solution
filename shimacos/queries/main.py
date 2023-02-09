import sys
import logging
from typing import Optional, Dict, Any

import hydra
from omegaconf import DictConfig
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(".")
from queries.bq import BQClient

logger = logging.getLogger()


def read_sql(path: str) -> str:
    with open(path, "r") as f:
        sql = "".join(f.readlines())
    return sql


def render_template(sql_path: str, params: Optional[Dict[str, Any]] = None) -> str:
    """jinjaテンプレートクエリをレンダリングする

    Args:
        sql_path (str): queryへの相対 or 絶対path
        params (Optional[Dict[str, Any]], optional): 置換したいパラメータの辞書. Defaults to None.

    Returns:
        str: レンダリングされたクエリ
    """
    if params is None:
        params = {}
    query_template = Template(read_sql(sql_path))
    query = query_template.render(params)
    return query


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    if config.execute_all:
        bq = BQClient(config.env.project_id)
        jobs = []
        executor = ThreadPoolExecutor()
        for version, sql_names in config.pipeline1.items():
            config.version = version
            for sql_name in sql_names:
                config.params.sql_name = sql_name
                bq.create_dataset(config.params.dataset_id)
                query = render_template(
                    f"./queries/{version}/{sql_name}.sql", config.params
                )
                logger.info(query)
                jobs.append(executor.submit(bq.execute_query, query))
        for future in as_completed(jobs):
            future.result()
            jobs.remove(future)

        for version in ["067", "069"]:
            config.version = version
            config.params.sql_name = "user_item"
            bq.create_dataset(config.params.dataset_id)
            query = render_template(f"./queries/{version}/user_item.sql", config.params)
            logger.info(query)
            jobs.append(executor.submit(bq.execute_query, query))
        for future in as_completed(jobs):
            future.result()
            jobs.remove(future)

        for version, sql_names in config.pipeline2.items():
            config.version = version
            for sql_name in sql_names:
                config.params.sql_name = sql_name
                bq.create_dataset(config.params.dataset_id)
                query = render_template(
                    f"./queries/{version}/{sql_name}.sql", config.params
                )
                logger.info(query)
                jobs.append(executor.submit(bq.execute_query, query))
        for future in as_completed(jobs):
            future.result()
            jobs.remove(future)

        for version in ["067", "069"]:
            for sql_name in ["dataset", "sampled_dataset"]:
                config.version = version
                config.params.sql_name = sql_name
                bq.create_dataset(config.params.dataset_id)
                query = render_template(
                    f"./queries/{version}/{sql_name}.sql", config.params
                )
                logger.info(query)
                jobs.append(executor.submit(bq.execute_query, query))

    else:
        query = render_template(
            f"./queries/{config.version}/{config.sql_name}.sql", config.params
        )
        logger.info(query)
        bq = BQClient(config.env.project_id)
        bq.create_dataset(config.params.dataset_id)
        bq.execute_query(query)


if __name__ == "__main__":
    main()
