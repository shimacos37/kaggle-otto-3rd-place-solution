{% for dataset_name in ['train_valid', 'train_test'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`

  AS

  WITH base AS (
    SELECT
      session,
      aid,
      type,
      ts
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
  ),

  session_max_ts AS (
    SELECT
      session,
      max(ts) AS max_ts
    FROM base
    GROUP BY 1
  ),

  session_rank AS (
    SELECT
      session,
      aid,
      type,
      timestamp_diff(max_ts, ts, SECOND) AS seconds_to_latest,
      row_number() OVER(PARTITION BY session, type ORDER BY ts DESC) AS rank
    FROM base
    LEFT JOIN session_max_ts USING (session)
  ),

  type_rank AS (
    SELECT
      session,
      aid,
      rank_clicks AS latest_rank_clicks,
      rank_carts AS latest_rank_carts,
      rank_orders AS latest_rank_orders
    FROM (SELECT
      session,
      aid,
      rank,
      type
      FROM session_rank)
    -- 最上位のみ取り出す
    PIVOT (min(rank) AS rank FOR type IN ("clicks", "carts", "orders"))
  ),

  type_seconds AS (
    SELECT *
    FROM (SELECT
      session,
      aid,
      seconds_to_latest,
      type
      FROM session_rank)
    -- 最上位のみ取り出す
    PIVOT (min(seconds_to_latest) AS seconds_to_latest FOR type IN ("clicks", "carts", "orders"))
  )

  SELECT *
  FROM type_rank
  LEFT JOIN type_seconds USING (session, aid);
{% endfor %}
