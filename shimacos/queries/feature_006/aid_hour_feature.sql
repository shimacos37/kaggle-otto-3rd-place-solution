{% for dataset_name in ['train_valid', 'train_test'] %}      

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`

  AS

  WITH base AS (
    SELECT
      session,
      aid,
      type,
      timestamp_trunc(ts, HOUR) AS ts_hour
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
  ),

  hour_count AS (
    SELECT *
    FROM (
      SELECT
        aid,
        ts_hour,
        type
      FROM base
    )
    PIVOT (count(*) AS hour_visit_count FOR type IN ("clicks", "carts", "orders"))
  ),

  hour_feature AS (
    SELECT
      aid,
      ts_hour,
      hour_visit_count_clicks,
      hour_visit_count_carts,
      hour_visit_count_orders,
      {% for hour in [1, 3, 5, 12] %}
        avg(coalesce(hour_visit_count_clicks, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS avg_{{hour}}hour_visit_count_clicks,
        stddev(coalesce(hour_visit_count_clicks, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS std_{{hour}}hour_visit_count_clicks,
        avg(coalesce(hour_visit_count_carts, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS avg_{{hour}}hour_visit_count_carts,
        stddev(coalesce(hour_visit_count_carts, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS std_{{hour}}hour_visit_count_carts,
        avg(coalesce(hour_visit_count_orders, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS avg_{{hour}}hour_visit_count_orders,
        stddev(coalesce(hour_visit_count_orders, 0)) OVER(
          PARTITION BY aid
          ORDER BY unix_seconds(ts_hour)
          RANGE BETWEEN {{hour * 3600}} PRECEDING AND CURRENT ROW
        ) AS std_{{hour}}hour_visit_count_orders{% endfor %}
    FROM unnest(generate_timestamp_array('{{start_ts}}', '{{end_ts}}', INTERVAL 1 HOUR)) AS ts_hour
    LEFT JOIN hour_count USING (ts_hour)
  )

  SELECT * FROM hour_feature;
      {% endfor %}
