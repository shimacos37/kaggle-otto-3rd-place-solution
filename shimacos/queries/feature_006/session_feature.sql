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
  )

  SELECT
    session,
    timestamp_diff(max(ts), min(ts), SECOND) AS search_secnonds,
    count(*) AS session_size,
    count(DISTINCT if(type = "clicks", aid, null)) AS click_aid_session_count,
    count(DISTINCT if(type = "carts", aid, null)) AS cart_aid_session_count,
    count(DISTINCT if(type = "orders", aid, null)) AS order_aid_session_count
  FROM base
  GROUP BY 1;
{% endfor %}
