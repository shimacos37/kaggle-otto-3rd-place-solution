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
      row_number() OVER(PARTITION BY session ORDER BY ts) AS rn
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    LEFT JOIN session_ts USING (session)
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
      count_orders
    FROM (
      SELECT
        aid_x,
        aid_y,
        type
      FROM pairs
    )
    PIVOT (count(*) AS count FOR type IN ("clicks", "carts", "orders"))
  ),

  cvr AS (
    SELECT
      aid_x,
      aid_y,
      (coalesce(count_carts, 0) + coalesce(count_orders, 0)) / (coalesce(count_clicks, 0) + avg_count_clicks) AS cvr_click_to_cart_order,
      coalesce(count_orders, 0) / (coalesce(count_carts, 0) + avg_count_carts) AS cvr_cart_to_order
    FROM type_counts,
      (SELECT
        avg(count_clicks) AS avg_count_clicks,
        avg(count_carts) AS avg_count_carts
        FROM type_counts)
  ),

  counts_cvr AS (
    SELECT *
    FROM type_counts
    LEFT JOIN cvr USING (aid_x, aid_y)
  ),

  covisit AS (
    SELECT *
    FROM (
      SELECT DISTINCT *
      FROM (
        SELECT
          aid_x,
          aid_y
        FROM counts_cvr
        WHERE count_clicks > 1
        QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_clicks DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC) <= 300
        UNION ALL
        SELECT
          aid_x,
          aid_y
        FROM counts_cvr
        WHERE count_carts > 1
        QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_carts DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC) <= 300
        UNION ALL
        SELECT
          aid_x,
          aid_y
        FROM counts_cvr
        WHERE count_orders > 1
        QUALIFY row_number() OVER(PARTITION BY aid_x ORDER BY count_orders DESC, cvr_click_to_cart_order DESC, cvr_cart_to_order DESC) <= 300
      )
    ) LEFT JOIN counts_cvr USING (aid_x, aid_y)
  ),

  latest_aids AS (
    SELECT
      session,
      aid,
      weight,
      pow(2, rn / max(rn) OVER(PARTITION BY session)) - 1 AS time_weight1
    FROM (
      SELECT
        session,
        aid,
        CASE
          WHEN type = "clicks" THEN 1
          WHEN type = "carts" THEN 6
          WHEN type = "orders" THEN 3
        END AS weight,
        row_number() OVER(PARTITION BY session ORDER BY ts DESC) AS rn
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
      LEFT JOIN session_ts USING (session)
      WHERE split != "train"
    )
  )



  SELECT
    session,
    aid_y AS aid,
    ----------
    -- type
    ----------
    sum(count_clicks * weight) AS sum_{{ sql_name }}_type_count_clicks,
    sum(count_carts * weight) AS sum_{{ sql_name }}_type_count_carts,
    sum(count_orders * weight) AS sum_{{ sql_name }}_type_count_orders,
    avg(count_clicks * weight) AS avg_{{ sql_name }}_type_count_clicks,
    avg(count_carts * weight) AS avg_{{ sql_name }}_type_count_carts,
    avg(count_orders * weight) AS avg_{{ sql_name }}_type_count_orders,
    stddev(count_clicks * weight) AS stddev_{{ sql_name }}_type_count_clicks,
    stddev(count_carts * weight) AS stddev_{{ sql_name }}_type_count_carts,
    stddev(count_orders * weight) AS stddev_{{ sql_name }}_type_count_orders,
    sum(weight) AS covisit_{{ sql_name }}_type_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_clicks * weight) DESC) AS rn_covisit_{{ sql_name }}_type_click,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_carts * weight) DESC) AS rn_covisit_{{ sql_name }}_type_cart,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_orders * weight) DESC) AS rn_covisit_{{ sql_name }}_type_order,
    row_number() OVER(PARTITION BY session ORDER BY sum(weight) DESC) AS rn_covisit_{{ sql_name }}_type_aid_count,
    ----------
    -- raw
    ----------
    sum(count_clicks) AS sum_{{ sql_name }}_count_clicks,
    sum(count_carts) AS sum_{{ sql_name }}_count_carts,
    sum(count_orders) AS sum_{{ sql_name }}_count_orders,
    avg(count_clicks) AS avg_{{ sql_name }}_count_clicks,
    avg(count_carts) AS avg_{{ sql_name }}_count_carts,
    avg(count_orders) AS avg_{{ sql_name }}_count_orders,
    stddev(count_clicks) AS stddev_{{ sql_name }}_count_clicks,
    stddev(count_carts) AS stddev_{{ sql_name }}_count_carts,
    stddev(count_orders) AS stddev_{{ sql_name }}_count_orders,
    sum(cvr_click_to_cart_order) AS sum_{{ sql_name }}_cvr_click_to_cart_order,
    sum(cvr_cart_to_order) AS sum_{{ sql_name }}_cvr_cart_to_order,
    avg(cvr_click_to_cart_order) AS avg_{{ sql_name }}_cvr_click_to_cart_order,
    avg(cvr_cart_to_order) AS avg_{{ sql_name }}_cvr_cart_to_order,
    stddev(cvr_click_to_cart_order) AS stddev_{{ sql_name }}_cvr_click_to_cart_order,
    stddev(cvr_cart_to_order) AS stddev_{{ sql_name }}_cvr_cart_to_order,
    count(*) AS covisit_{{ sql_name }}_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_clicks) DESC) AS rn_covisit_{{ sql_name }}_click,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_carts) DESC) AS rn_covisit_{{ sql_name }}_cart,
    row_number() OVER(PARTITION BY session ORDER BY sum(count_orders) DESC) AS rn_covisit_{{ sql_name }}_order,
    row_number() OVER(PARTITION BY session ORDER BY sum(weight) DESC) AS rn_covisit_{{ sql_name }}_aid_count,
    row_number() OVER(PARTITION BY session ORDER BY sum(cvr_click_to_cart_order) DESC) AS rn_covisit_{{ sql_name }}_cvr_click_to_cart_order,
    row_number() OVER(PARTITION BY session ORDER BY sum(cvr_cart_to_order) DESC) AS rn_covisit_{{ sql_name }}_cvr_cart_to_order
  FROM latest_aids AS l
  LEFT JOIN covisit AS r
    ON l.aid = r.aid_x
  WHERE aid_y IS NOT NULL
  GROUP BY 1, 2
  QUALIFY
    rn_covisit_{{ sql_name }}_click <= 1000
    OR rn_covisit_{{ sql_name }}_cart <= 1000
    OR rn_covisit_{{ sql_name }}_order <= 1000
    OR rn_covisit_{{ sql_name }}_aid_count <= 1000
    OR rn_covisit_{{ sql_name }}_type_click <= 1000
    OR rn_covisit_{{ sql_name }}_type_cart <= 1000
    OR rn_covisit_{{ sql_name }}_type_order <= 1000
    OR rn_covisit_{{ sql_name }}_type_aid_count <= 1000
    OR rn_covisit_{{ sql_name }}_cvr_click_to_cart_order <= 1000
    OR rn_covisit_{{ sql_name }}_cvr_cart_to_order <= 1000;

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_candidates`

  AS

  SELECT * FROM `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}_feature`
  WHERE
    rn_covisit_{{ sql_name }}_click <= 100
    OR rn_covisit_{{ sql_name }}_cart <= 100
    OR rn_covisit_{{ sql_name }}_order <= 100
    OR rn_covisit_{{ sql_name }}_aid_count <= 100
    OR rn_covisit_{{ sql_name }}_type_click <= 100
    OR rn_covisit_{{ sql_name }}_type_cart <= 100
    OR rn_covisit_{{ sql_name }}_type_order <= 100
    OR rn_covisit_{{ sql_name }}_type_aid_count <= 100
    OR rn_covisit_{{ sql_name }}_cvr_click_to_cart_order <= 100
    OR rn_covisit_{{ sql_name }}_cvr_cart_to_order <= 100;

{% endfor %}
