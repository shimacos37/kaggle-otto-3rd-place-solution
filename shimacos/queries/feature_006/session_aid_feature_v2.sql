{% for dataset_name in ['train_valid', 'train_test'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`

  AS

  WITH base AS (
    SELECT
      session,
      aid,
      type,
      ts,
      dense_rank() OVER(PARTITION BY session ORDER BY ts DESC) AS session_dense_rank
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
      session_dense_rank,
      timestamp_diff(max_ts, ts, SECOND) AS seconds_to_latest
    FROM base
    LEFT JOIN session_max_ts USING (session)
  ),

  session_count AS (
    SELECT
      session,
      aid,
      visit_count_clicks,
      visit_count_carts,
      visit_count_orders
    FROM (
      SELECT
        session,
        aid,
        type
      FROM base
    )
    PIVOT (count(*) AS visit_count FOR type IN ("clicks", "carts", "orders"))
  ),

  time_weighted_session_count AS (
    SELECT
      session,
      aid,
      time_weighted_visit_count_clicks,
      time_weighted_visit_count_carts,
      time_weighted_visit_count_orders
    FROM (
      SELECT
        session,
        aid,
        type,
        1 / (ln(seconds_to_latest + 1) + 1) AS weight
      FROM session_rank
    )
    PIVOT (sum(weight) AS time_weighted_visit_count FOR type IN ("clicks", "carts", "orders"))
  ),

  time_feature AS (
    SELECT
      session,
      aid,
      min(seconds_to_latest) AS min_seconds_to_latest,
      max(seconds_to_latest) AS max_seconds_to_latest,
      avg(session_dense_rank) AS avg_session_dense_rank,
      stddev(session_dense_rank) AS std_session_dense_rank
    FROM session_rank
    GROUP BY 1, 2
  )


  SELECT DISTINCT
    *,
    visit_count_clicks + visit_count_carts * 6 + visit_count_orders * 3 AS type_weighted_visit_counts,
    row_number() OVER(
      PARTITION BY session
      ORDER BY visit_count_clicks + visit_count_carts * 6 + visit_count_orders * 3 DESC
    ) AS rn_type_weighted_visit_counts
  FROM session_count
  LEFT JOIN time_weighted_session_count USING (session, aid)
  LEFT JOIN time_feature USING (session, aid);
{% endfor %}
