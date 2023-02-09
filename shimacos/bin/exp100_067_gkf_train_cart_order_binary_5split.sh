

docker-compose run --rm dev \
    python main_lgbm.py \
    feature.version=067 \
    lgbm.n_fold=5 \
    lgbm.params.objective=binary \
    lgbm.params.metric=binary_logloss \
    feature.label_cols=['click_label','cart_order_label'] \
    test=False \
    store.model_name=$(basename $0 .sh) 

docker-compose run --rm dev \
    python check_cv.py \
    lgbm.n_fold=5 \
    feature.version=067 \
    store.model_name=$(basename $0 .sh) 

docker-compose run --rm dev \
    python predict.py \
    lgbm.n_fold=5 \
    feature.version=067 \
    store.model_name=$(basename $0 .sh) 
