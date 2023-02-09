{% for dataset_name in ['train_valid', 'train_test'] %}      

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`

  AS

  WITH base AS (
    SELECT
      session,
      aid,
      type,
      lag(type) OVER(PARTITION BY session, aid ORDER BY ts) AS lag_type,
      timestamp_trunc(ts, DAY) AS ts_day
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
  ),

  repeat_feature AS (
    SELECT
      aid,
      {% for fn in ['avg', 'stddev', 'min', 'max'] %}
        {{ fn }}(repeat_count) AS {{fn}}_repeat_count{% endfor %}
    FROM (
      SELECT
        session,
        aid,
        count(*) AS repeat_count
      FROM base
      GROUP BY 1, 2
    )
    GROUP BY 1
  ),

  transition_feature AS (
    SELECT *
    FROM (
      SELECT
        aid,
        concat(lag_type, "_", type) AS transition,
        transition_count / (sum(transition_count) OVER(PARTITION BY aid, lag_type) + avg(transition_count) OVER()) AS transition_prob
      FROM (
        SELECT
          aid,
          lag_type,
          type,
          count(*) AS transition_count
        FROM base
        GROUP BY 1, 2, 3
      )
    )
    PIVOT (
      sum(
        transition_prob
      ) AS transition_prob FOR transition IN ("clicks_clicks", "clicks_carts", "clicks_orders", "carts_clicks", "carts_carts", "carts_orders", "orders_clicks", "orders_carts", "orders_orders")
    )
  )

  SELECT *
  FROM repeat_feature
  FULL OUTER JOIN transition_feature USING (aid);
{% endfor %}
