{% for dataset_name in ['train_valid', 'train_test'] %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`


  AS

  {% if dataset_name == 'train_valid' %}
{% set split_ts = '2022-08-14T22:00:00'%}
{% else %}
{% set split_ts = '2022-08-21T22:00:00' %}
{% endif %}

  WITH session_ts AS (
    SELECT
      session,
      min(ts) AS min_ts
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    GROUP BY 1
  ),

  last_aids AS (
    SELECT *
    FROM (
      SELECT
        session,
        aid AS last_click_aid
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
      WHERE type = "clicks"
      QUALIFY row_number() OVER(PARTITION BY session ORDER BY ts DESC) = 1
    )
    FULL OUTER JOIN (
      SELECT
        session,
        aid AS last_cart_aid
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
      WHERE type = "carts"
      QUALIFY row_number() OVER(PARTITION BY session ORDER BY ts DESC) = 1
    ) USING (session)
    FULL OUTER JOIN (
      SELECT
        session,
        aid AS last_order_aid
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
      WHERE type = "orders"
      QUALIFY row_number() OVER(PARTITION BY session ORDER BY ts DESC) = 1
    ) USING (session)
  )

  SELECT *
  FROM (
    SELECT
      session,
      neighbor_aid AS aid,
      cosine_similarity[offset(i)] AS last_click_cosine_similarity
    FROM last_aids AS l
    LEFT JOIN `otto_preprocess_003.w2v_64_min_count5_window5_sg_cosine_similarity_{{ dataset_name }}` AS r
      ON l.last_click_aid = r.aid,
      unnest(neighbor_aids) AS neighbor_aid WITH OFFSET AS i
    WHERE aid != neighbor_aid
  )
  FULL OUTER JOIN (
    SELECT
      session,
      neighbor_aid AS aid,
      cosine_similarity[offset(i)] AS last_cart_cosine_similarity
    FROM last_aids AS l
    LEFT JOIN `otto_preprocess_003.w2v_64_min_count5_window5_sg_cosine_similarity_{{ dataset_name }}` AS r
      ON l.last_cart_aid = r.aid,
      unnest(neighbor_aids) AS neighbor_aid WITH OFFSET AS i
    WHERE aid != neighbor_aid
  ) USING (session, aid)
  FULL OUTER JOIN (
    SELECT
      session,
      neighbor_aid AS aid,
      cosine_similarity[offset(i)] AS last_order_cosine_similarity
    FROM last_aids AS l
    LEFT JOIN `otto_preprocess_003.w2v_64_min_count5_window5_sg_cosine_similarity_{{ dataset_name }}` AS r
      ON l.last_order_aid = r.aid,
      unnest(neighbor_aids) AS neighbor_aid WITH OFFSET AS i
    WHERE aid != neighbor_aid
  ) USING (session, aid)
  WHERE
    last_click_cosine_similarity >= 0.85
    OR
    last_cart_cosine_similarity >= 0.85
    OR
    last_order_cosine_similarity >= 0.85;
{% endfor %}
