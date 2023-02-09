CREATE OR REPLACE TABLE `{{ project_id }}.otto_valid_result.makotu_v3`

AS

WITH
{% for type in ['click', 'cart', 'order'] %}
  {{type}}_pred AS (
    SELECT *
    FROM `{{ project_id }}.otto_valid_result.{{ type }}_makotu_v3`
  
  )
  {% if not loop.last %}
    ,
  {% endif %}
{% endfor %}


SELECT * FROM click_pred
FULL OUTER JOIN cart_pred USING (session, aid)
FULL OUTER JOIN order_pred USING (session, aid);

CREATE OR REPLACE TABLE `{{ project_id }}.otto_test_result.makotu_v3`

AS

WITH
{% for type in ['click', 'cart', 'order'] %}
  {{type}}_pred AS (
    SELECT *
    FROM `{{ project_id }}.otto_test_result.{{ type }}_makotu_v3`
  
  )
  {% if not loop.last %}
    ,
  {% endif %}
{% endfor %}


SELECT * FROM click_pred
FULL OUTER JOIN cart_pred USING (session, aid)
FULL OUTER JOIN order_pred USING (session, aid)
