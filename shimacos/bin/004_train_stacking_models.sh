
# Prepare dataset
docker-compose run --rm dev python queries/main.py version=stacking_010 sql_name=train_dataset,test_dataset -m

# Train and predict
sh bin/exp108_stacking_010.sh
sh bin/exp109_stacking_010_catboost.sh