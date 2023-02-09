

docker-compose run --rm gpu \
    python stacking_cat.py \
    feature.version=010 \
    store.model_name=$(basename $0 .sh) 
