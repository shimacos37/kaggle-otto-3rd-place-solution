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
      lead(aid) OVER(PARTITION BY session ORDER BY ts) AS lead_aid,
      lead(type) OVER(PARTITION BY session ORDER BY ts) AS lead_type,
      row_number() OVER(PARTITION BY session ORDER BY ts) AS rn
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    WHERE True
    QUALIFY lead_aid IS NOT NULL
  ),

  pairs AS (
    SELECT
      aid AS aid_x,
      lead_aid AS aid_y,
      count(*) AS transition_count,
      count(DISTINCT session) AS transition_user_count
    FROM base
    GROUP BY 1, 2
  ),

  aid_count AS (
    SELECT
      aid_x AS aid,
      sum(transition_count) AS sum_transition_count,
      sum(transition_user_count) AS sum_transition_user_count
    FROM pairs
    GROUP BY 1
  ),

  cvr_count AS (
    SELECT
      aid_x,
      aid_y,
      transition_count / sqrt(coalesce(r1.sum_transition_count, 1) * coalesce(r2.sum_transition_count, 1)) AS norm_transition_count,
      transition_user_count / sqrt(coalesce(r1.sum_transition_user_count, 1) * coalesce(r2.sum_transition_user_count, 1)) AS norm_transition_user_count
    FROM pairs
    LEFT JOIN aid_count AS r1
      ON pairs.aid_x = r1.aid
    LEFT JOIN aid_count AS r2
      ON pairs.aid_y = r2.aid
  ),

  covisit AS (
    SELECT *
    FROM (
      SELECT DISTINCT *
      FROM (
        SELECT
          aid_x,
          aid_y
        FROM cvr_count
        WHERE True
        QUALIFY
          row_number() OVER(PARTITION BY aid_x ORDER BY norm_transition_count DESC) <= 500
          OR
          row_number() OVER(PARTITION BY aid_x ORDER BY norm_transition_user_count DESC) <= 500
      )
    )
    LEFT JOIN cvr_count USING (aid_x, aid_y)
  ),

  latest_aids AS (
    SELECT
      session,
      aid,
      type_weight,
      pow(2, rn / max(rn) OVER(PARTITION BY session)) - 1 AS time_weight1
    FROM (
      SELECT
        session,
        aid,
        CASE
          WHEN type = "clicks" THEN 1
          WHEN type = "carts" THEN 6
          WHEN type = "orders" THEN 3
        END AS type_weight,
        row_number() OVER(PARTITION BY session ORDER BY ts DESC) AS rn
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
      WHERE split != "train"
    )
  )



  SELECT
    session,
    aid_y AS aid,
    ----------
    -- type
    ----------
    sum(norm_transition_count * type_weight) AS sum_{{ sql_name }}_type_weight_norm_transition_count,
    sum(norm_transition_user_count * type_weight) AS sum_{{ sql_name }}_type_weight_norm_transition_user_count,
    avg(norm_transition_count * type_weight) AS avg_{{ sql_name }}_type_weight_norm_transition_count,
    avg(norm_transition_user_count * type_weight) AS avg_{{ sql_name }}_type_weight_norm_transition_user_count,
    stddev(norm_transition_count * type_weight) AS stddev_{{ sql_name }}_type_weight_norm_transition_count,
    stddev(norm_transition_user_count * type_weight) AS stddev_{{ sql_name }}_type_weight_norm_transition_user_count,
    sum(type_weight) AS {{ sql_name }}_type_weight_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(norm_transition_count * type_weight) DESC) AS rn_{{ sql_name }}_type_weight_norm_transition_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(norm_transition_user_count * type_weight) DESC) AS rn_{{ sql_name }}_type_weight_norm_transition_user_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(type_weight) DESC) AS rn_{{ sql_name }}_type_weight_aid_count,
    ----------
    -- raw
    ----------
    sum(norm_transition_count) AS sum_{{ sql_name }}_norm_transition_count,
    sum(norm_transition_user_count) AS sum_{{ sql_name }}_norm_transition_user_count,
    avg(norm_transition_count) AS avg_{{ sql_name }}_norm_transition_count,
    avg(norm_transition_user_count) AS avg_{{ sql_name }}_norm_transition_user_count,
    stddev(norm_transition_count) AS stddev_{{ sql_name }}_norm_transition_count,
    stddev(norm_transition_user_count) AS stddev_{{ sql_name }}_norm_transition_user_count,
    count(*) AS {{ sql_name }}_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(norm_transition_count) DESC) AS rn_{{ sql_name }}_norm_transition_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(norm_transition_user_count) DESC) AS rn_{{ sql_name }}_norm_transition_user_count,
    row_number() OVER(PARTITION BY session ORDER BY count(*) DESC) AS rn_{{ sql_name }}_aid_count
  FROM latest_aids AS l
  LEFT JOIN covisit AS r
    ON l.aid = r.aid_x
  WHERE aid_y IS NOT NULL
  GROUP BY 1, 2
  QUALIFY
    rn_{{ sql_name }}_type_weight_norm_transition_count <= 1000
    OR rn_{{ sql_name }}_type_weight_norm_transition_user_count <= 1000
    OR rn_{{ sql_name }}_type_weight_aid_count <= 1000
    OR rn_{{ sql_name }}_norm_transition_count <= 1000
    OR rn_{{ sql_name }}_norm_transition_user_count <= 1000
    OR rn_{{ sql_name }}_aid_count <= 1000;

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_candidates`

  AS

  SELECT * FROM `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`
  WHERE
    rn_{{ sql_name }}_type_weight_norm_transition_count <= 50
    OR rn_{{ sql_name }}_type_weight_norm_transition_user_count <= 50
    OR rn_{{ sql_name }}_type_weight_aid_count <= 50
    OR rn_{{ sql_name }}_norm_transition_count <= 50
    OR rn_{{ sql_name }}_norm_transition_user_count <= 50
    OR rn_{{ sql_name }}_aid_count <= 50;

{% endfor %}
