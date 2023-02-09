

docker-compose run --rm dev \
    python stacking_lgbm.py \
    feature.version=010 \
    store.model_name=$(basename $0 .sh) test=False

docker-compose run --rm dev \
    python stacking_lgbm.py \
    feature.version=010 \
    store.model_name=$(basename $0 .sh) test=True
