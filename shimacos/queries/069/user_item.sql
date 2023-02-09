{% for dataset_name in ['train_valid', 'train_test'] %}      

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ dataset_name }}_{{ sql_name }}`
  CLUSTER BY session

  AS

  WITH
  sampled_session AS (
    SELECT DISTINCT session FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    WHERE split != "train"
  )

  SELECT DISTINCT
    session,
    aid,
    split
  FROM (
      {% for candidate in [
                          "click2click_covisit_1hour",
                          "covisit_items_window_size5_v3",
                          "covisit_bigram_v2",
                          "covisit_items_1days",
                        ]
      %}
        SELECT
          session,
          aid
        FROM `{{ project_id }}.otto_candidates_018.{{ candidate }}_{{ dataset_name }}_candidates`
        UNION ALL
      {% endfor %}
      SELECT DISTINCT
        session,
        aid
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    )
  LEFT JOIN (
      SELECT DISTINCT
        session,
        split
      FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    ) USING (session)
  WHERE session IN (SELECT session FROM sampled_session);
      {% endfor %}
