import pandas as pd
import numpy as np
import pickle

from load_preprocess import file_loader




def load_make_val_predictions(train_model_path = 'train_model', val_pickle_path = 'prepr_val')
with open('train_model', 'rb') as f:
    model = pickle.load(f)
    
X_val = pd.read_pickle(val_pickle_path)
X_val['predict_proba'] = [model.predict_proba([x])[:,1][0] for x in tqdm(X_val['text_large_combined'])]


def advises(df, client_id):
    uncklean_list = df.loc[client_id,'text_split']
    clean_list = df.loc[client_id,'text_split_cleaned']
    res = []
    for ul, cl in zip(uncklean_list, clean_list):
        if df.loc[client_id,'predict_proba'] > 0.1:
            if len(cl) > 50:
                pred_proba = mp.predict_proba([cl])[:,1][0]
                if pred_proba > 0.4:
                    res.append([pred_proba, ul])
        if len(res)>0:
            advise_df = pd.DataFrame(res).sort_values(by=0, ascending = False).drop_duplicates()\
            .assign(client_id = client_id)
            return advise_df
        else :
            return 
        pd.DataFrame()