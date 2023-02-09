# Alvor's part of the OTTO Kaggle Competition solution. <br> 0.599 Public LB Model


<img width="956" alt="Screenshot 2023-02-08 at 18 33 02" src="https://user-images.githubusercontent.com/41992707/217607600-c35eb73f-46e6-4f6f-bf59-5c576fecd5ae.png">

<img width="443" alt="Screenshot 2023-02-08 at 18 33 59" src="https://user-images.githubusercontent.com/41992707/217607743-d32fa8a0-a926-4ce1-98a4-a3ff016b9367.png">

## 1. Get input data.

### 1.1. Get Competition Data. <br>Place three public Kaggle Datasets into the **"alvor/input"** folder of current repository:

1. [OTTO Chunk Data in Parquet Format](https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format) by [colum2131
](https://www.kaggle.com/columbia2131)

2. [otto-validation](https://www.kaggle.com/datasets/cdeotte/otto-validation) by [Chris Deotte](https://www.kaggle.com/cdeotte)

3. [Otto Full Optimized Memory Footprint](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint) by [Radek Osmulski](https://www.kaggle.com/radek1)

Directory structure should look like this:

<img width="335" alt="tree-1" src="https://user-images.githubusercontent.com/41992707/215167246-85bb0e01-a17c-4b88-80ef-f68f05a6b9be.png">

### 1.2. Get Feature Engineering Data. 

#### 1.2.1.
Place the files with co-visitation matrices for training and validation
```
cm_30_30_012_012_0_v11m_0.pqt
cm_30_30_012_012_0_v21k_0.pqt
cm_30_30_012_012_0_v21m_0.pqt
cm_30_30_012_012_3_v31m_0.pqt
cm_30_30_012_12_0_v51ha_0.pqt
```
from the **Version-10** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottomatrices) into the **“alvor/matrices”** folder of the repository

(If you want to reproduce these files, you need to run **Version-12** of the [Kaggle notebook](https://www.kaggle.com/code/allvor/co-matrices-maker))

#### 1.2.2.

Place the files with co-visitation matrices for prediction
```
cm_30_30_012_012_0_v11m_1.pqt
cm_30_30_012_012_0_v21k_1.pqt
cm_30_30_012_012_0_v21m_1.pqt
cm_30_30_012_012_3_v31m_1.pqt
cm_30_30_012_12_0_v51ha_1.pqt
```
from the **Version-11** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottomatrices) into the **“alvor/matrices”** folder of the repository

(If you want to reproduce these files, you need to run **Version-13** of the [Kaggle notebook](https://www.kaggle.com/code/allvor/co-matrices-maker))

#### 1.2.3.

Place the file with matrix factorization embeddings 
```
emb_32_1_sh1_pub.npy
```
from the **Version-3** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoshiftsoutput/versions/3) 
into the **"alvor/matrices"** folder of the repository.

(If you want to reproduce this file, you need to run **Version-3** of the [Kaggle notebook](https://www.kaggle.com/code/allvor/ottoshiftskernel/notebook);
and it needs **Version-1** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoshifts) to run)

#### 1.2.4.

Place the file with matrix factorization embeddings
```
emb_32_1_sh2_pub.npy
```
from the **Version-6** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoshiftsoutput/versions/6) 
into the **"alvor/matrices"** folder of the repository.

(If you want to reproduce this file, you need to run **Version-6** of the [Kaggle notebook](https://www.kaggle.com/code/allvor/ottoshiftskernel/notebook);
and it needs **Version-3** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoshifts) to run)

#### 1.2.5.

Place the file with word2vec embeddings
```
w2v.npy
```
from the **Version-1** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottow2vembs/versions/1)
into the **"alvor/matrices"** folder of the repository.

(If you want to reproduce this file, you need to run **Version-3** of the [Kaggle notebook](https://www.kaggle.com/code/allvor/word2vec-model-training-and-submission-0-533))


#### 1.2.6.

Place the file with word2vec embeddings
```
w2v_100.npy
```
from the **Version-1** of the [Dataset](https://www.kaggle.com/datasets/allvor/alvorottow2v100)
into the **"alvor/matrices"** folder of the repository.

(If you want to reproduce this file, you need to run the local Jupyter notebook **alvor/w2v.ipynb** from the current repository.


## 5. Feature engineering

Run **twice** each of the respository notebooks
```
FE_sessions.ipynb
FE_items.ipynb
FE_aids2sessions.ipynb
```
as follows: 

one time with option **MODE = 0** in the second cell 

and one time with option **MODE = 1** in the second sell.

As a result, output files will be placed into the "feats" folder of the repository:
```
FE_sessions_0.pqt
FE_sessions_1.pqt
FE_aids_0.pqt
FE_aids_1.pqt
FE_aids2sessions_0.pqt
FE_aids2sessions_1.pqt
```

## 6. Prepare training and prediction dataframes

Run the Jupyter Notebook from the current repository:
```
make_train_df.ipynb
``` 
As a result, the output files will be placed into the "feats" folder of the repository:
```
feats_0_batch_0.pqt
feats_0_batch_0_small.pqt
feats_0_batch_1.pqt
feats_0_batch_1_small.pqt
feats_0_batch_2.pqt
feats_0_batch_2_small.pqt
feats_0_batch_3.pqt
feats_0_batch_3_small.pqt
feats_0_batch_0.pqt
feats_0_batch_1.pqt
feats_0_batch_2.pqt
feats_0_batch_3.pqt
```
Now the repository's directory structure should look like this:

<img width="350" alt="Screenshot 2023-02-01 at 13 27 42" src="https://user-images.githubusercontent.com/41992707/216042483-86129206-417e-496c-8ff8-b01ca10e4583.png">

## 7. Train models for .596 submission

### 7.1. Train models

**(!) You already have trained models files in "models" folder if you don't want to re-train them. Otherwise:**

Run two Jupyter notebooks from current repository: 
```
train_model_clicks.ipynb
train_model_carts_orders.ipynb
``` 
As a result, output files will be placed into the "models" folder of the repository.
<br> At this moment the "models" folder should look like this:

<img width="486" alt="Screenshot 2023-02-01 at 15 03 55" src="https://user-images.githubusercontent.com/41992707/216064455-5342551b-6a4e-41ef-8b05-5f5c16e29b0d.png">

(file names could contain slightly different numbers)

### 7.2. Prepare raw predictions and submission

Run Jupyter Notebook
```
make_submission.ipynb 
```
to get submission which scores 0.596 on Public LB:
```
submission.csv
```
raw predictions file for blending purposes (You can download it from **Version-3** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoalvorraw/versions/3)):
```
alvor_raw_predictions_596.parquet
```
out-of-folds files for blending purposes (You can download them from **Version-1** of the [Dataset](https://www.kaggle.com/datasets/allvor/otto-alvor-oofs/versions/1)):
```
alvor_oof_clicks.parquet
alvor_oof_carts_orders.parquet
```
and a file with the list of my candidates (You can download it from **Version-1** of the [Dataset](https://www.kaggle.com/datasets/allvor/alvorcandidatesbig)):
```
alvor_candidates_big.parquet
```

## 8. Add BPR features and Bigram features from my teammate @sirius81 to improve my model 0.596 -> 0.599

### 8.1.

[sirius](https://www.kaggle.com/sirius81) has code to calculate BPR features and Bigram features, so I provided him with 2 files from the previous section:
```
alvor_raw_predictions_596.parquet
alvor_candidates_big.parquet
```
and he was so kind to calculate those features for me.

### 8.2.

Place **"bpr"** and **"test_bpr"** folders from Version-2 of the [Dataset](https://www.kaggle.com/datasets/sirius81/otto-bprembedding) by [sirius](https://www.kaggle.com/sirius81) into the "matrices" folder of the repository.

Place **"alvor_bigram"** and **"test_alvor_bigram"** folders from Version-4 of the [Dataset](https://www.kaggle.com/datasets/sirius81/otto-features) by [sirius](https://www.kaggle.com/sirius81) into the "matrices" folder of the repository.

Directory structure should look like this now:

<img width="456" alt="Screenshot 2023-02-01 at 15 33 00" src="https://user-images.githubusercontent.com/41992707/216071761-1666a571-56b1-4198-b328-ee1e21edc056.png">

### 8.3.

Run Jupyter Notebook to prepare [sirius](https://www.kaggle.com/sirius81) features for further use:
```
handle_sirius_features.ipynb
```
As a result, some output files will be added to the "matrices" folder:

<img width="341" alt="Screenshot 2023-02-01 at 17 04 30" src="https://user-images.githubusercontent.com/41992707/216096968-f95d68fa-e6f4-44b4-a87b-e14eaf68af16.png">

## 9. Train models for .599 submission

### 9.1. Train models

**(!) You already have trained models files in "models_new" folder if you don't want to re-train them. Otherwise:**

Run two Jupyter notebooks from current repository: 
```
train_model_clicks_two.ipynb
train_model_carts_orders_two.ipynb
``` 
As a result, output files will be placed into the "models_new" folder of the repository.
<br> At this moment the "models_new" folder should look like this:

<img width="447" alt="Screenshot 2023-02-01 at 17 32 58" src="https://user-images.githubusercontent.com/41992707/216104290-3ab1f09f-9770-4ec6-a98a-f459d12e4a75.png">

(file names could contain slightly different numbers)

### 9.2. Prepare raw predictions and submission

Run Jupyter Notebook
```
make_submission_two.ipynb 
```
to get submission which scores 0.599 on Public LB:
```
submission2.csv
```
raw predictions file for blending purposes (You can download it from **Version-4** of the [Dataset](https://www.kaggle.com/datasets/allvor/ottoalvorraw/versions/4)):
```
alvor_raw_predictions_599.parquet
```
and out-of-folds files for blending purposes (You can download them from **Version-2** of the [Dataset](https://www.kaggle.com/datasets/allvor/otto-alvor-oofs/versions/2)):
```
alvor_oof_clicks_v2.parquet
alvor_oof_carts_orders_v2.parquet
```


### 10. Acknowledgements

In my model I use code of some public Kaggle Notebooks. I am grateful to the authors of these notebooks for their work and for making it public:

[Candidate Rerank Model](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575) by [Chris Deotte](https://www.kaggle.com/cdeotte)

[Matrix Factorization with GPU](https://www.kaggle.com/code/cpmpml/matrix-factorization-with-gpu) by [CPMP](https://www.kaggle.com/cpmpml)

[Word2Vec Model](https://www.kaggle.com/code/balaganiarz0/word2vec-model-training-and-submission-0-533) by [Jakub Gorski](https://www.kaggle.com/balaganiarz0)



