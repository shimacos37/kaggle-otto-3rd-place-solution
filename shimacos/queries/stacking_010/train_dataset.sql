
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
    session_ids as (
      {% for model_name in all_model_names %}
      SELECT DISTINCT session FROM `{{ project_id }}.otto_valid_result.{{model_name}}`
      {% if not loop.last %}
      INTERSECT DISTINCT
      {% endif %}
      {% endfor %}
    ),
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
              session,
              aid,
              click_pred_score[OFFSET(rank)] as {{model_name}}_pred_clicks,
              rank + 1 as {{model_name}}_clicks_rank
            FROM `{{ project_id }}.otto_valid_result.{{model_name}}`,
            UNNEST(click_pred) as aid with offset as rank
          )
          FULL OUTER JOIN (
            SELECT
              session,
              aid,
              cart_pred_score[OFFSET(rank)] as {{model_name}}_pred_carts,
              rank + 1 as {{model_name}}_carts_rank
            FROM `{{ project_id }}.otto_valid_result.{{model_name}}`,
            UNNEST(cart_pred) as aid with offset as rank
          ) USING(session, aid)
          FULL OUTER JOIN (
            SELECT
              session,
              aid,
              order_pred_score[OFFSET(rank)] as {{model_name}}_pred_orders,
              rank + 1 as {{model_name}}_orders_rank
            FROM `{{ project_id }}.otto_valid_result.{{model_name}}`,
            UNNEST(order_pred) as aid with offset as rank
          ) USING(session, aid)
          WHERE 
            session IN (SELECT * FROM session_ids)
            AND (
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
        FROM `{{ project_id }}.otto_valid_result.all_click_makotu_v3`
        WHERE session IN (SELECT * FROM session_ids)
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
        * EXCEPT(
          click_label, click_labels, cart_label, order_label
        ),
        cast(aid IN unnest(cart_label) AS int64) AS cart_label,
        cast(aid IN unnest(order_label) AS int64) AS order_label,
        coalesce(cast(aid = click_label AS int64), 0) AS click_label
      FROM pred
      LEFT JOIN `{{ project_id }}.otto_preprocess_003.ground_truth` 
      USING(session)
      WHERE mod(session, 10) IN (0, 1)
