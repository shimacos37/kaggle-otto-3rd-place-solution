CREATE OR REPLACE TABLE `{{ project_id }}.otto_valid_result.makotu_v4`

AS

WITH
{% for type in ['cart', 'order'] %}
  {{type}}_pred AS (
    SELECT *
    FROM `{{ project_id }}.otto_valid_result.{{ type }}_makotu_v4`
  
  )
  {% if not loop.last %}
    ,
  {% endif %}
{% endfor %}


SELECT * FROM cart_pred
FULL OUTER JOIN order_pred USING (session, aid);

CREATE OR REPLACE TABLE `{{ project_id }}.otto_test_result.makotu_v4`

AS

WITH
{% for type in ['cart', 'order'] %}
  {{type}}_pred AS (
    SELECT *
    FROM `{{ project_id }}.otto_test_result.{{ type }}_makotu_v4`
  
  )
  {% if not loop.last %}
    ,
  {% endif %}
{% endfor %}


SELECT * FROM cart_pred
FULL OUTER JOIN order_pred USING (session, aid)
