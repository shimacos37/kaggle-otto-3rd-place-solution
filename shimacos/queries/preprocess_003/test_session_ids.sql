{% for i in range(10) %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ i }}`

  AS

  SELECT DISTINCT session FROM `{{ project_id }}.{{ dataset_id }}.train_test`
  WHERE split = "test" AND mod(session, 10) = {{i}};
{% endfor %}
