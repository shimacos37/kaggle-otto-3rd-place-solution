

# BPR
docker-compose run --rm gpu python scripts/bpr.py

# Word2Vec
docker-compose run --rm gpu python scripts/word2vec.py

# Node2Vec
docker-compose run --rm gpu python scripts/node2vec.py

# SQL
docker-compose run --rm dev python queries/main.py execute_all=True