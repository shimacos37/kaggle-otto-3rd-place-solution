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

  n_session_feature AS (
    -- 期間中全てのアクションから計算
    SELECT
      aid,
      n_session_clicks,
      n_session_carts,
      n_session_orders
    FROM (
      SELECT
        aid,
        type,
        count(DISTINCT session) AS n_session
      FROM base
      GROUP BY 1, 2
    )
    PIVOT (sum(n_session) AS n_session FOR type IN ("clicks", "carts", "orders"))
  ),

  count_feature AS (
    -- 期間中全てのアクションから計算
    SELECT
      aid,
      action_count_clicks,
      action_count_carts,
      action_count_orders
    FROM (
      SELECT
        aid,
        type,
        count(*) AS action_count
      FROM base
      GROUP BY 1, 2
    )
    PIVOT (sum(action_count) AS action_count FOR type IN ("clicks", "carts", "orders"))
  ),

  time_feature AS (
    -- dataset作成時に使う
    -- 最後のアクションからどれくらい時間が経ってるか
    -- 最初のアクションからどれくらい時間が経ってるか
    SELECT
      aid,
      min(ts) AS action_min_ts,
      max(ts) AS action_max_ts
    FROM base
    GROUP BY 1
  ),

  cvr_feature AS (
    SELECT
      aid,
      (n_session_carts + n_session_orders) / (n_session_clicks + avg(n_session_clicks) OVER()) AS n_session_cvr_click_to_cart_order,
      (action_count_carts + action_count_orders) / (action_count_clicks + avg(action_count_clicks) OVER()) AS action_count_cvr_click_to_cart_order,
      (n_session_carts) / (n_session_clicks + avg(n_session_clicks) OVER()) AS n_session_cvr_click_to_cart,
      (action_count_carts) / (action_count_clicks + avg(action_count_clicks) OVER()) AS action_count_cvr_click_to_cart,
      (n_session_orders) / (n_session_clicks + avg(n_session_clicks) OVER()) AS n_session_cvr_click_to_order,
      (action_count_orders) / (action_count_clicks + avg(action_count_clicks) OVER()) AS action_count_cvr_click_to_order,
      (n_session_orders) / (n_session_carts + avg(n_session_carts) OVER()) AS n_session_cvr_cart_to_order,
      (action_count_orders) / (action_count_carts + avg(action_count_carts) OVER()) AS action_count_cvr_cart_to_order
    FROM n_session_feature
    LEFT JOIN count_feature USING (aid)
  )

  SELECT *
  FROM n_session_feature
  LEFT JOIN count_feature USING (aid)
  LEFT JOIN time_feature USING (aid)
  LEFT JOIN cvr_feature USING (aid);
{% endfor %}
