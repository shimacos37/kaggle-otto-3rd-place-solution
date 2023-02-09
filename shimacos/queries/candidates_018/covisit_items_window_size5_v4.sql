{% for dataset_name in ['train_valid', 'train_test'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`

{% if dataset_name == 'train_valid' %}
{% set split_ts = '2022-08-14T22:00:00'%}
{% else %}
{% set split_ts = '2022-08-21T22:00:00' %}
{% endif %}

  AS

  WITH session_ts AS (
    SELECT
      session,
      min(ts) AS min_ts
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    GROUP BY 1
  ),

  base AS (
    SELECT
      session,
      aid,
      type,
      ts,
      dense_rank() OVER(PARTITION BY session ORDER BY ts) AS rn
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    LEFT JOIN session_ts USING(session)
    WHERE True
    QUALIFY DENSE_RANK() OVER(PARTITION BY session ORDER BY ts DESC) <= 30
),

  pairs AS (
    SELECT DISTINCT
      a.session,
      a.aid AS aid_x,
      b.aid AS aid_y,
      b.type
    FROM base AS a
    LEFT JOIN base AS b ON a.session = b.session
  WHERE
    a.rn != b.rn
    AND
    abs(b.rn - a.rn) <= 2
), 

type_counts AS (
  SELECT
    aid_x,
    aid_y,
    count_clicks,
    count_carts,
    count_orders,
    count_clicks + count_carts + count_orders as count_all,
    count_clicks + count_carts * 6 + count_orders * 3 as weighted_count_all,
  FROM (
    SELECT
      aid_x,
      aid_y,
      type
    FROM pairs
  )
  PIVOT (count(*) AS count FOR type IN ("clicks", "carts", "orders"))
),

 cvr as (
  select
    aid_x,
    aid_y,
    (ifnull(count_carts, 0) + ifnull(count_orders, 0)) / (ifnull(count_clicks, 0) + avg_count_clicks) as cvr_click_to_cart_order,
    ifnull(count_orders, 0) / (ifnull(count_carts, 0) + avg_count_carts) as cvr_cart_to_order,
  from type_counts,
  (select avg(count_clicks) as avg_count_clicks, avg(count_carts) as avg_count_carts from type_counts)
),

counts_cvr as (
  select *
  from type_counts
  left join cvr using(aid_x, aid_y)
),

covisit as (
  SELECT
    *
  FROM (
    SELECT DISTINCT *
    FROM (
      SELECT
        aid_x,
        aid_y,
      FROM counts_cvr
      WHERE count_clicks > 1
      QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_clicks DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC)  <= 300
      UNION ALL
      SELECT
        aid_x,
        aid_y,
      FROM counts_cvr
      WHERE count_carts > 1
      QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_carts DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC)  <= 300
      UNION ALL
      SELECT
        aid_x,
        aid_y,
      FROM counts_cvr
      WHERE count_orders > 1
      QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_orders DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC)  <= 300
      UNION ALL
      SELECT
        aid_x,
        aid_y,
      FROM counts_cvr
      WHERE count_all > 1
      QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_all DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC)  <= 300
      UNION ALL
      SELECT
        aid_x,
        aid_y,
      FROM counts_cvr
      WHERE weighted_count_all > 1
      QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY weighted_count_all DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC)  <= 300
    )
  ) LEFT JOIN counts_cvr USING(aid_x, aid_y)
),

latest_aids as (
  SELECT
    session,
    aid,
    weight,
    pow(2, rn / max(rn) over(partition by session)) - 1 as time_weight1,
  FROM (
    SELECT
      session,
      aid,
      CASE
        WHEN type = "clicks" THEN 1
        WHEN type = "carts" THEN 6
        WHEN type = "orders" THEN 3
      END AS weight,
      row_number() over(partition by session order by ts desc) as rn,
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    LEFT JOIN session_ts USING(session)
    WHERE split != 'train'
  )
)



  SELECT
    session,
    aid_y AS aid,
    ----------
    -- type
    ----------
    {% for _count in ['count_clicks', 'count_carts', 'count_orders', 'count_all', 'weighted_count_all']%}
      {% for fn in ['sum', 'avg', 'stddev'] %}
      {{fn}}({{_count}} * weight) AS {{fn}}_{{sql_name}}_type_{{_count}},
      {{fn}}({{_count}}) AS {{fn}}_{{sql_name}}_{{_count}},
      {% endfor %}
      ROW_NUMBER() OVER(PARTITION BY session ORDER BY sum({{_count}}) DESC) AS rn_covisit_{{sql_name}}_{{_count}},
      ROW_NUMBER() OVER(PARTITION BY session ORDER BY sum({{_count}} * weight) DESC) AS rn_covisit_{{sql_name}}_type_{{_count}},
    {% endfor %}

      sum(weight) AS covisit_{{sql_name}}_type_count,
      row_number() OVER(PARTITION BY session ORDER BY sum(weight) DESC) AS rn_covisit_{{sql_name}}_type_aid_count,
    ----------
    -- raw
    ----------
    sum(cvr_click_to_cart_order) as sum_{{sql_name}}_cvr_click_to_cart_order,
    sum(cvr_cart_to_order) as sum_{{sql_name}}_cvr_cart_to_order,
    avg(cvr_click_to_cart_order) as avg_{{sql_name}}_cvr_click_to_cart_order,
    avg(cvr_cart_to_order) as avg_{{sql_name}}_cvr_cart_to_order,
    stddev(cvr_click_to_cart_order) as stddev_{{sql_name}}_cvr_click_to_cart_order,
    stddev(cvr_cart_to_order) as stddev_{{sql_name}}_cvr_cart_to_order,
    count(*) AS covisit_{{sql_name}}_count,
    row_number() OVER(PARTITION BY session ORDER BY count(*) DESC) AS rn_covisit_{{sql_name}}_aid_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(cvr_click_to_cart_order) DESC) AS rn_covisit_{{sql_name}}_cvr_click_to_cart_order,
    row_number() OVER(PARTITION BY session ORDER BY sum(cvr_cart_to_order) DESC) AS rn_covisit_{{sql_name}}_cvr_cart_to_order,
  FROM latest_aids AS l
  LEFT JOIN covisit AS r
    ON l.aid = r.aid_x
  WHERE aid_y IS NOT NULL
  GROUP BY 1, 2
  QUALIFY
      {% for _count in ['count_clicks', 'count_carts', 'count_orders', 'count_all', 'weighted_count_all']%}
      rn_covisit_{{sql_name}}_type_{{_count}} <= 1000
      OR rn_covisit_{{sql_name}}_{{_count}} <= 1000
      OR
      {% endfor %}
      rn_covisit_{{sql_name}}_type_aid_count <= 1000
      OR rn_covisit_{{sql_name}}_aid_count <= 1000
      OR rn_covisit_{{sql_name}}_cvr_click_to_cart_order <= 1000
      OR rn_covisit_{{sql_name}}_cvr_cart_to_order <= 1000

;

CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_candidates`

AS

SELECT * FROM `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`
WHERE
      {% for _count in ['count_clicks', 'count_carts', 'count_orders', 'count_all', 'weighted_count_all']%}
      rn_covisit_{{sql_name}}_type_{{_count}} <= 100
      OR rn_covisit_{{sql_name}}_{{_count}} <= 100
      OR
      {% endfor %}
      rn_covisit_{{sql_name}}_type_aid_count <= 100
      OR rn_covisit_{{sql_name}}_aid_count <= 100
      OR rn_covisit_{{sql_name}}_cvr_click_to_cart_order <= 100
      OR rn_covisit_{{sql_name}}_cvr_cart_to_order <= 100

;

{% endfor %}
