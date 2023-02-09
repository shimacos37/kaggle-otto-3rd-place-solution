import hydra
import numpy as np
import pandas as pd
from jinja2 import Template
from omegaconf import DictConfig


@hydra.main(config_path="../yamls", config_name="sql.yaml")
def main(config: DictConfig) -> None:
    model_names = [
        "exp108_stacking_010",
        "exp109_stacking_010_catboost",
    ]
    type_weights = {
        "click": [2, 1],
        "cart": [1, 1],
        "order": [1, 1],
    }
    query = """
    WITH
    {% for model_name in model_names %}
      {{model_name}} as (
        SELECT
          session_type,
          aid,
          1 / (rank + 1) as rank,
        FROM `{{project_id}}.otto_test_result.{{model_name}}` ,
        UNNEST(SPLIT(labels, ' ')) as aid with offset as rank
        WHERE rank <= 49 AND CONTAINS_SUBSTR(session_type, '{{type}}')
    ),
    {% endfor%}
      BASE AS (
        SELECT DISTINCT
          *
        FROM (
          {% for model_name in model_names %}
          SELECT session_type, aid FROM {{model_name}}
          {% if not loop.last %}
          UNION ALL
          {% endif %}
          {% endfor %}
        )
      ),
    popularity as (
      select
        CAST(aid AS STRING) as aid,
        row_number() over(order by n_session_{{type}}s desc) as rn,
      from otto_feature_002.aid_feature_train_test
      qualify rn <= 20
    ), pred as (
      SELECT
        session_type,
        ARRAY_AGG(aid ORDER BY rank DESC) as labels,
        ARRAY_LENGTH(ARRAY_AGG(aid ORDER BY rank DESC)) as length,
      FROM (
        SELECT
          session_type,
          aid,
          (
              {% for model_name, weight in model_name_weights %}
              IFNULL({{model_name}}.rank, 1 / 50) * {{weight}}
              {% if not loop.last %}
              +
              {% endif %}
              {% endfor %}
          ) / ({{num_model}}) as rank,
        FROM BASE
        {% for model_name in model_names %}
        FULL OUTER JOIN {{model_name}} USING(session_type, aid)
        {% endfor %}
        WHERE session_type != 'session_type'
        QUALIFY
          ROW_NUMBER() OVER(PARTITION BY session_type ORDER BY rank DESC) <= 20
      )
      GROUP BY 1
    ), interpolate as (
      select
        session_type,
        array_agg(aid order by rn) as interpolate_aids,
      from pred, popularity
      where rn <= 20 - length
      group by 1
    )
    SELECT
      session_type,
      ARRAY_TO_STRING(if(interpolate_aids is null, labels, array_concat(labels, interpolate_aids)), " ") as labels,
    FROM pred
    LEFT JOIN interpolate USING(session_type)
    ORDER BY 1
    """
    dfs = []
    for type, weights in type_weights.items():
        rendered_query = Template(query).render(
            {
                "project_id": config.env.project_id,
                "model_names": model_names,
                "num_model": len(model_names),
                "model_name_weights": zip(model_names, weights),
                "type": type,
            }
        )
        df = pd.read_gbq(
            rendered_query,
            project_id=config.env.project_id,
            use_bqstorage_api=True,
            progress_bar_type="tqdm",
        )
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv("./output/blending/v16.csv", index=False)


if __name__ == "__main__":
    main()
