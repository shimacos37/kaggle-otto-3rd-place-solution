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

  WITH bpr_feature AS (
    SELECT
      session,
      aid,
      dot_product(session_vector, aid_vector) AS bpr_distance
    FROM `{{ project_id }}.otto_067.{{ dataset_name }}_user_item`
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.bpr_user_vectors_{{ dataset_name }}`
      USING (session)
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.bpr_aid_vectors_{{ dataset_name }}`
      USING (aid)
  ),


  diff_feature AS (
    SELECT
      session,
      aid,
      bpr_distance - avg_bpr_distance AS diff_avg_bpr_distance,
      bpr_distance - min_bpr_distance AS diff_min_bpr_distance,
      bpr_distance - max_bpr_distance AS diff_max_bpr_distance
    FROM bpr_feature
    LEFT JOIN (
      SELECT
        session,
        avg(bpr_distance) AS avg_bpr_distance,
        min(bpr_distance) AS min_bpr_distance,
        max(bpr_distance) AS max_bpr_distance
      FROM bpr_feature
      GROUP BY 1
    ) USING (session)
  )

  SELECT *
  FROM bpr_feature
  LEFT JOIN diff_feature USING (session, aid);
{% endfor %}
