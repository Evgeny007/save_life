import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


def model_trainer(X, y, train_df_path = 'prepr_train',model_path = 'predict_model'):
    """ Train prediction model and save to disk as pickle file"""
    
    train_df = pd.read_pickle(train_df_path)
    
    assert 'text_large_combined' in X.columns
    mp = make_pipeline(TfidfVectorizer(analyzer = 'word', ngram_range = (1,1), min_df = 20, max_df = 0.2), 
                         SGDClassifier(loss='log', n_jobs=-1, early_stopping  =True, class_weight='balanced'))# ComplementNB()
    mp.fit(X['text_large_combined'],y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(mp, f)