import pandas as pd
import numpy as np
import random, os, gc
import pickle

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def gc_clear():
    for i in range(5):
        gc.collect()
        
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def evaluate_candidates(candidates):
    denom = {
        'clicks' : 1755534,
        'carts' : 576482,
        'orders' : 313303
    }
    
    test_labels = pd.read_parquet(
        'input/otto-validation/test_labels.parquet'
    ).explode('ground_truth')
    test_labels['ground_truth'] = test_labels['ground_truth'].astype(int)
    test_labels.columns = ['session','type','aid']
    s = test_labels.merge(
        candidates,
        on=['session','aid'],
        how='inner'
    )
    s['pred'] = 1
    s['n'] = s.groupby(
        ['session','type']
    )['pred'].rank(method='first')
    s = s[s['n']<21].copy()
    
    return [
        np.round(
            s[s['type']==target]['pred'].sum() 
            / denom[target],
            6
        )
        for target in ['clicks','carts','orders']
    ]

def prepare_submission(s, prediction, target, version):
    s['pred'] = prediction
    s = s.sort_values(
        ['session','pred'],
        ascending = [True,False]
    )
    s = s.reset_index(drop=True)
    s['n'] = s.groupby('session').cumcount()
    s = s[s['n']<20][['session','aid']].copy()
    sub = pd.DataFrame(
        s.groupby(['session'])['aid'].apply(list)
    ).reset_index()
    sub['labels'] = [' '.join(map(str, l)) for l in sub['aid']]
    sub['session'] = sub['session'].astype(str) + f"_{target}"
    sub.columns = ['session_type','aid','labels']
    sub[['session_type','labels']].to_parquet(
        f'output/sub_{target}_lgbm_v{version}.pqt',index=False
    )
    del s
    del sub
    gc_clear()
    
def get_best_feats(FIS, N):
    out_dict = dict()

    for f in FIS:
        target = f.split("_")[2]

        df = pd.read_csv(
            f"models/{f}"
        ).groupby(
            'feature',
        ).agg(
            {'importance2':'mean'}
        ).sort_values(
            by=['importance2'],
            ascending=False
        ).reset_index()

        out_dict[target] = list(df['feature'][:N])

    full_feats = []
    for x in out_dict:
        full_feats.extend(out_dict[x])
    full_feats = sorted(list(set(full_feats)))
    
    return out_dict, full_feats