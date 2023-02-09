{% for dataset_name in ['train_valid'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ dataset_name }}_{{ sql_name }}`
  CLUSTER BY session, fold

  AS

  WITH base AS (
    SELECT * FROM `{{ project_id }}.{{ dataset_id }}.{{ dataset_name }}_dataset`
    WHERE split != "train"
  ),

  folds AS (
    SELECT
      session,
      mod(row_number() OVER(ORDER BY rand()), 5) AS fold
    FROM (SELECT DISTINCT session FROM base)
  ),

  positive AS (
    SELECT *
    FROM base
    WHERE click_label = 1 OR cart_label = 1 OR order_label = 1
  ),

  positive_count AS (
    SELECT count(*) AS positive_size
    FROM positive
  ),

  negative AS (
    SELECT base.*
    FROM base, positive_count
    WHERE
      click_label != 1 AND cart_label != 1 AND order_label != 1
    QUALIFY row_number() OVER(ORDER BY rand()) <= 20 * positive_size
  )

  SELECT
    *,
    cast(cart_label + order_label >= 1 AS INT64) AS cart_order_label
  FROM (
      SELECT * FROM positive
      UNION ALL
      SELECT * FROM negative
    )
  LEFT JOIN folds USING (session);
{% endfor %}
