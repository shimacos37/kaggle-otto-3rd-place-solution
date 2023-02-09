
  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}`

  AS
  {% set model_names = [
        "makotu_v3",
        "makotu_v4",
        "alvor_raw_predictions_596",
        "alvor_raw_predictions_599",
        "weighted_order_cart",
  ] %}
  {% set shimacos_model_names = [
        "exp100_067_gkf_train_cart_order_binary_5split",
        "exp105_069_without_w2b_bpr_candidats",
  ] %}
  {% set all_model_names = [
        "makotu_v3",
        "makotu_v4",
        "alvor_raw_predictions_596",
        "alvor_raw_predictions_599",
        "weighted_order_cart",
        "exp100_067_gkf_train_cart_order_binary_5split",
        "exp105_069_without_w2b_bpr_candidats",
  ] %}
  
  WITH 
      {% for model_name in model_names %}
        {{model_name}} as (
          SELECT
            session,
            aid,
            {% if model_name != 'makotu_v4' %}
            pred_clicks as {{model_name}}_pred_clicks,
            {% endif %}
            pred_carts as {{model_name}}_pred_carts,
            pred_orders as {{model_name}}_pred_orders,
            {% if model_name != 'makotu_v4' %}
            DENSE_RANK() OVER(PARTITION BY session ORDER BY pred_clicks DESC) as {{model_name}}_clicks_rank,
            {% endif %}
            DENSE_RANK() OVER(PARTITION BY session ORDER BY pred_carts DESC) as {{model_name}}_carts_rank,
            DENSE_RANK() OVER(PARTITION BY session ORDER BY pred_orders DESC) as {{model_name}}_orders_rank,
          FROM `{{project_id}}.otto_test_result.{{model_name}}`
          WHERE True
          QUALIFY 
            {{model_name}}_orders_rank <= 50
            OR
            {{model_name}}_carts_rank <= 50
            {% if model_name != 'makotu_v4' %}
            OR
            {{model_name}}_clicks_rank <= 50
            {% endif %}
      ),
      {% endfor %}
      {% for model_name in shimacos_model_names %}
        {{model_name}} as (
          SELECT
            *
          FROM (
            SELECT
              CAST(SPLIT(session_type, '_')[OFFSET(0)] AS INT64) as session,
              CAST(aid AS INT64) as aid,
              scores[OFFSET(rank)] as {{model_name}}_pred_clicks,
              rank + 1 as {{model_name}}_clicks_rank
            FROM `{{ project_id }}.otto_test_result.{{model_name}}`,
            UNNEST(SPLIT(labels, " ")) as aid with offset as rank
            WHERE CONTAINS_SUBSTR(session_type, 'click') AND aid != '<NA>'
          )
          FULL OUTER JOIN (
            SELECT
              CAST(SPLIT(session_type, '_')[OFFSET(0)] AS INT64) as session,
              CAST(aid AS INT64) as aid,
              scores[OFFSET(rank)] as {{model_name}}_pred_carts,
              rank + 1 as {{model_name}}_carts_rank
            FROM `{{ project_id }}.otto_test_result.{{model_name}}`,
            UNNEST(SPLIT(labels, " ")) as aid with offset as rank
            WHERE CONTAINS_SUBSTR(session_type, 'cart') AND aid != '<NA>'
          ) USING(session, aid)
          FULL OUTER JOIN (
            SELECT
              CAST(SPLIT(session_type, '_')[OFFSET(0)] AS INT64) as session,
              CAST(aid AS INT64) as aid,
              scores[OFFSET(rank)] as {{model_name}}_pred_orders,
              rank + 1 as {{model_name}}_orders_rank
            FROM `{{ project_id }}.otto_test_result.{{model_name}}`,
            UNNEST(SPLIT(labels, " ")) as aid with offset as rank
            WHERE CONTAINS_SUBSTR(session_type, 'order') AND aid != '<NA>'
          ) USING(session, aid)
          WHERE 
            (
              {{model_name}}_clicks_rank <= 50
              OR
              {{model_name}}_carts_rank <= 50
              OR
              {{model_name}}_orders_rank <= 50
            )
      ),
      {% endfor %}
      makotu_v3_click_all as (
        select
          session,
          aid,
          pred_all_clicks as makotu_v3_pred_clicks_all,
          ROW_NUMBER() OVER(PARTITION BY session ORDER BY pred_all_clicks DESC) as makotu_v3_clicks_rank_all,
        FROM `{{ project_id }}.otto_test_result.all_click_makotu_v3`
        QUALIFY 
          makotu_v3_clicks_rank_all <= 50
      ),
      base AS (
        SELECT DISTINCT
          *
        FROM (
          {% for model_name in all_model_names %}
            SELECT session, aid FROM {{model_name}}
            UNION ALL
          {% endfor %}
            SELECT session, aid FROM makotu_v3_click_all
        )
      ),
      pred AS (
        SELECT
          *
        FROM base
        {% for model_name in all_model_names %}
        LEFT JOIN {{model_name}} USING(session, aid)
        {% endfor %}
        LEFT JOIN makotu_v3_click_all USING(session, aid)
      )
      SELECT
        *
      FROM pred
