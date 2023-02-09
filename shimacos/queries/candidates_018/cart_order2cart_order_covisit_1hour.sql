{% for dataset_name in ['train_valid', 'train_test'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`

  {% if dataset_name == 'train_valid' %}
{% set split_ts = '2022-08-14T22:00:00'%}
{% else %}
{% set split_ts = '2022-08-21T22:00:00' %}
{% endif %}

  AS

  WITH

  base AS (
    SELECT
      session,
      aid,
      type,
      ts,
      row_number() OVER(PARTITION BY session ORDER BY ts) AS rn
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    WHERE
      type IN ("carts", "orders")
  ),


  pairs AS (
    SELECT DISTINCT
      -- sessionユニークに取ってるので、どれだけ色々なsessionで同時購入されているかが重要
      a.session,
      a.aid AS aid_x,
      b.aid AS aid_y
    FROM base AS a
    LEFT JOIN base AS b ON a.session = b.session
    WHERE
      a.rn != b.rn
      AND
      abs(timestamp_diff(a.ts, b.ts, SECOND)) < 1 * 60 * 60
  ),

  type_counts AS (
    SELECT
      aid_x,
      aid_y,
      count(*) AS {{ sql_name }}_weight
    FROM pairs
    GROUP BY 1, 2
    HAVING {{ sql_name }}_weight > 1
  ),

  latest_aids AS (
    SELECT
      session,
      aid,
      CASE
        WHEN type = "clicks" THEN 1
        WHEN type = "carts" THEN 6
        WHEN type = "orders" THEN 3
      END AS weight,
      pow(2, percent_rank() OVER(PARTITION BY session ORDER BY ts) + 0.1) - 1 AS time_weight
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    WHERE split != "train"
  )

  SELECT
    session,
    aid_y AS aid,
    sum({{ sql_name }}_weight) AS sum_{{ sql_name }}_weight,
    avg({{ sql_name }}_weight) AS avg_{{ sql_name }}_weight,
    count(*) AS covisit_{{ sql_name }}_count,
    sum({{ sql_name }}_weight * weight) AS sum_{{ sql_name }}_type_weight,
    avg({{ sql_name }}_weight * weight) AS avg_{{ sql_name }}_type_weight,
    sum(weight) AS covisit_{{ sql_name }}_type_count,
    sum({{ sql_name }}_weight * time_weight) AS sum_{{ sql_name }}_time_weight,
    avg({{ sql_name }}_weight * time_weight) AS avg_{{ sql_name }}_time_weight,
    sum(time_weight) AS covisit_{{ sql_name }}_time_count,
    sum({{ sql_name }}_weight * time_weight * weight) AS sum_{{ sql_name }}_type_time_weight,
    avg({{ sql_name }}_weight * time_weight * weight) AS avg_{{ sql_name }}_type_time_weight,
    sum(time_weight * weight) AS covisit_{{ sql_name }}_type_time_count,
    row_number() OVER(PARTITION BY session ORDER BY sum({{ sql_name }}_weight) DESC) AS rn_{{ sql_name }},
    row_number() OVER(PARTITION BY session ORDER BY sum({{ sql_name }}_weight * time_weight) DESC) AS rn_time_weight_{{ sql_name }},
    row_number() OVER(PARTITION BY session ORDER BY sum({{ sql_name }}_weight * weight) DESC) AS rn_type_weight_{{ sql_name }},
    row_number() OVER(PARTITION BY session ORDER BY sum({{ sql_name }}_weight * time_weight * weight) DESC) AS rn_type_time_weight_{{ sql_name }}
  FROM latest_aids AS l
  LEFT JOIN type_counts AS r
    ON l.aid = r.aid_x
  GROUP BY 1, 2
  QUALIFY
    rn_{{ sql_name }} <= 1000
    OR rn_time_weight_{{ sql_name }} <= 1000
    OR rn_type_weight_{{ sql_name }} <= 1000
    OR rn_type_time_weight_{{ sql_name }} <= 1000;

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_candidates`

  AS

  SELECT * FROM `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`
  WHERE
    rn_{{ sql_name }} <= 30
    OR rn_time_weight_{{ sql_name }} <= 30
    OR rn_type_weight_{{ sql_name }} <= 30
    OR rn_type_time_weight_{{ sql_name }} <= 30;
{% endfor %}
