CREATE TEMP FUNCTION dot_product(vec1 array<float64>, vec2 array<float64>)
RETURNS float64
AS (
  (SELECT sum(elem1 * elem2)
    FROM unnest(vec1) AS elem1 WITH OFFSET AS pos1
    INNER JOIN unnest(vec2) AS elem2 WITH OFFSET AS pos2
      ON pos1 = pos2)
);

CREATE TEMP FUNCTION l2_distance(vec1 array<float64>, vec2 array<float64>)
RETURNS float64
AS (
  (SELECT sqrt(sum(pow(elem1 - elem2, 2)))
    FROM unnest(vec1) AS elem1 WITH OFFSET AS pos1
    INNER JOIN unnest(vec2) AS elem2 WITH OFFSET AS pos2
      ON pos1 = pos2)
);

{% for dataset_name in ['train_valid', 'train_test'] %}        

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ dataset_name }}`

  AS

  WITH last_aids AS (
    SELECT
      session,
      aid AS last_aid
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    WHERE type = "carts"
    QUALIFY
      row_number() OVER(PARTITION BY session ORDER BY ts DESC) = 1
  ),

  n2v_feature AS (
    SELECT
      session,
      l.aid,
      dot_product(v1.vector, v2.vector) AS last_cart_n2v_cosine_similarity,
      l2_distance(v1.vector, v2.vector) AS last_cart_n2v_l2_distance
    FROM `{{ project_id }}.otto_067.{{ dataset_name }}_user_item` AS l
    LEFT JOIN last_aids USING (session)
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.node2vec_128_all_vectors_{{ dataset_name }}` AS v1
      ON l.aid = v1.aid
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.node2vec_128_all_vectors_{{ dataset_name }}` AS v2
      ON last_aid = v2.aid
  ),

  w2v_feature AS (
    SELECT
      session,
      l.aid,
      dot_product(v1.vector, v2.vector) AS last_cart_w2v_cosine_similarity,
      l2_distance(v1.vector, v2.vector) AS last_cart_w2v_l2_distance
    FROM `{{ project_id }}.otto_067.{{ dataset_name }}_user_item` AS l
    LEFT JOIN last_aids USING (session)
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.w2v_64_min_count5_window5_sg_vector_{{ dataset_name }}` AS v1
      ON l.aid = v1.aid
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.w2v_64_min_count5_window5_sg_vector_{{ dataset_name }}` AS v2
      ON last_aid = v2.aid
  ),

  joined AS (
    SELECT *
    FROM n2v_feature
    LEFT JOIN w2v_feature USING (session, aid)
  ),

  diff_feature AS (
    SELECT
      session,
      aid,
      {% for distance in ['cosine_similarity', 'l2_distance',] %}
        last_cart_w2v_{{ distance }} - avg_w2v_{{ distance }} AS diff_avg_last_cart_w2v_{{distance}},
        last_cart_w2v_{{ distance }} - min_w2v_{{ distance }} AS diff_min_last_cart_w2v_{{distance}},
        last_cart_w2v_{{ distance }} - max_w2v_{{ distance }} AS diff_max_last_cart_w2v_{{distance}},
        last_cart_n2v_{{ distance }} - avg_n2v_{{ distance }} AS diff_avg_last_cart_n2v_{{distance}},
        last_cart_n2v_{{ distance }} - min_n2v_{{ distance }} AS diff_min_last_cart_n2v_{{distance}},
        last_cart_n2v_{{ distance }} - max_n2v_{{ distance }} AS diff_max_last_cart_n2v_{{distance}}{% endfor %}
    FROM joined
    LEFT JOIN (
      SELECT
        session,
        {% for distance in ['cosine_similarity', 'l2_distance'] %}
          avg(last_cart_w2v_{{ distance }}) AS avg_w2v_{{distance}},
          avg(last_cart_n2v_{{ distance }}) AS avg_n2v_{{distance}},
          min(last_cart_w2v_{{ distance }}) AS min_w2v_{{distance}},
          min(last_cart_n2v_{{ distance }}) AS min_n2v_{{distance}},
          max(last_cart_w2v_{{ distance }}) AS max_w2v_{{distance}},
          max(last_cart_n2v_{{ distance }}) AS max_n2v_{{distance}}{% endfor %}
      FROM joined
      GROUP BY 1
    ) USING (session)
  )

  SELECT *
  FROM joined
  LEFT JOIN diff_feature USING (session, aid);
        {% endfor %}
