{% for i in range(20) %}

  CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ sql_name }}_{{ i }}`

  AS

  SELECT DISTINCT session FROM `{{ project_id }}.{{ dataset_id }}.train_valid`
  WHERE split = "valid" AND mod(session, 20) = {{i}};
{% endfor %}
