{% for dataset_name in ['train_valid', 'train_test'] %}    

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ dataset_name }}_{{ sql_name }}`
  CLUSTER BY session, split

  AS

  WITH
  session_ts AS (
    SELECT
      session,
      timestamp_trunc(timestamp_add(max(ts), INTERVAL 1 SECOND), MINUTE) AS ts_minute,
      timestamp_trunc(timestamp_add(max(ts), INTERVAL 1 SECOND), HOUR) AS ts_hour,
      timestamp_trunc(timestamp_add(max(ts), INTERVAL 1 SECOND), DAY) AS ts_day
    FROM `{{ project_id }}.otto_preprocess_003.{{ dataset_name }}`
    GROUP BY 1
  ),

  joined AS (
    SELECT
      * EXCEPT(
        click_label, click_labels, cart_label, order_label, ts_hour, ts_day, action_min_ts, action_max_ts, ts_minute
      ),
      cast(aid IN unnest(cart_label) AS int64) AS cart_label,
      cast(aid IN unnest(order_label) AS int64) AS order_label,
      timestamp_diff(ts_minute, action_max_ts, SECOND) AS seconds_from_last_aid_action,
      timestamp_diff(ts_minute, action_min_ts, SECOND) AS seconds_from_first_aid_action,
      coalesce(cast(aid = click_label AS int64), 0) AS click_label
    FROM (
      SELECT DISTINCT
        session,
        aid,
        split
      FROM `{{ project_id }}.{{ dataset_id }}.{{ dataset_name }}_user_item`
    )
    LEFT JOIN `{{ project_id }}.otto_preprocess_003.ground_truth` USING (session)
    LEFT JOIN session_ts USING (session)
    LEFT JOIN `{{ project_id }}.otto_feature_006.aid_hour_feature_{{ dataset_name }}` USING (aid, ts_hour)
    LEFT JOIN `{{ project_id }}.otto_feature_006.latest_user_items_feature_{{ dataset_name }}` USING (session, aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.aid_repeat_feature_{{ dataset_name }}` USING (aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_aid_feature_v3_{{ dataset_name }}` USING (session, aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_feature_{{ dataset_name }}` USING (session)
    LEFT JOIN `{{ project_id }}.otto_feature_006.aid_feature_{{ dataset_name }}` USING (aid)
    {% for candidate in [
                          "click2click_covisit_1hour",
                          "covisit_items_window_size5_v4",
                          "covisit_bigram_v2",
                          "covisit_items_1days",
                          "covisit_items_1hour",
                          "cart_order2cart_order_covisit_1hour"
                  ]
%}    
      LEFT JOIN `{{ project_id }}.otto_candidates_018.{{ candidate }}_{{ dataset_name }}_feature` USING (session, aid)
    {% endfor %}
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_aid_vector_feature_067_last_cart_{{ dataset_name }}` USING (session, aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_aid_vector_feature_067_last_order_{{ dataset_name }}` USING (session, aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_aid_vector_feature_067_last_{{ dataset_name }}` USING (session, aid)
    LEFT JOIN `{{ project_id }}.otto_feature_006.session_aid_vector_feature_067_bpr_{{ dataset_name }}` USING (session, aid)
  )

  SELECT * FROM joined;

    {% endfor %}
