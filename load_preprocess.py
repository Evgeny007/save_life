import pandas as pd
import numpy as np
from tqdm import tqdm
import re


def file_loader(path):
    """ For loading train and validation datasets"""
    df = pd.read_csv(path).dropna()
    return df

def regexp_text_cleaner(sentence):
    """ Clean the text: removes all but words"""
    
    pat_del = '([\(\)\/])|([\\d+])'
    pat_space = '[_\.]'
    pat_main = '[^ ]*[а-яёый\\\/]{1,}[$ ]*'
    
    sentence = ' '.join(re.findall(pat_main, sentence))
    clear_name = re.sub(pat_del,'',sentence)
    sentence = re.sub(pat_space, ' ',sentence)
    return sentence

def largest_text_info(text_in_list, n_largest = 1):
    """ From all sentences in patients scripts choose the lagest"""
    try:
        sorted_series = pd.Series(text_in_list).sort_values(key = lambda x: x.str.len(), ascending = False)
        result = sorted_series.iloc[n_largest-1]
    except IndexError:
        result = np.nan
    return result

def total_preprocessinag(df):
    df['text'] = df['text'].str.replace(',\n','\n').str.lower()
    df['text_split'] = df['text'].str.split('\n')
    df['text_split_cleaned'] = df['text_split'].apply(lambda x: [regexp_text_cleaner(i) for i in x])
    df['text_split_cleaned_full'] = df['text_split_cleaned'].str.join(' , ')
    
    df['text_1_large'] = df['text_split'].apply(lambda x: largest_text_info(x, n_largest=1))
    df['text_2_large'] = df['text_split'].apply(lambda x: largest_text_info(x, n_largest=2))
    df['text_3_large'] = df['text_split'].apply(lambda x: largest_text_info(x, n_largest=3))
    df['text_large_combined']  = df['text_1_large'] + ' ' + df['text_2_large']+ ' ' + df['text_3_large']
    df['text_large_combined'] = df['text_large_combined'].astype(str).apply(regexp_text_cleaner)   
    
    return df.drop(columns  = ['text_1_large','text_2_large','text_3_large'])

############################ find in text structured columns and add them to dataset ############

columns = ['температура тела :', 'веc, кг :','рост, см :','педикулез :','чесотка :','флюорограмма','гепатит',
           'вич (спид) :','хронические заболевания печени :','оперативные вмешательства :',
          'переливание крови :','контакты с инфекционными больными :','пребывание за границей :','лихорадка :','код по мкб10',
          'госпитализирован по поводу данного заболевания в текущем году :','порядок госпитализации :',
           'способ поступления (доставки) :','цель поступления :','канал поступления:']

def columns_extractor(text_list, ind,columns = columns):
    """ Extract chosen by hand column names from top 30 sentences of text"""
    res_df = pd.Series({x: y.replace(x,'') for x in columns for y in text_list[:30] if x in y}, name = ind)\
    .drop_duplicates().to_frame().T
    return res_df


def temperature_finder(x):
    """ Handle with temperature data"""
    x_splitted = x.strip().split()
    for i in x_splitted:
        for t in ['35','36','37','38']:
            if t in x:
                x_r = i.replace(',','.')
                return float(x_r)


def weight_height_finder(x):
    """ Handle with weight and height dtat"""
    x_splitted = x.strip().split()
    for i in x_splitted:
        try :
            res = int(float(x.replace(',','.').replace(' ','')))
            return res
        except Exception:
            return np.nan
        

def sep_columns_df_maker(df,columns = columns):
    """ For chosen column names presented in text in strucutured way"""
    
    list_df = [columns_extractor(df.loc[i,'text_split'], i, columns = columns) for i in tqdm(df.index, position=0)]
    res_df =pd.concat(list_df)
    res_df.columns = [x.replace(':','').strip() for x in res_df.columns]
    
    pattern_sub = '[A-zА-я]+'
    if 'температура тела' in res_df.columns:
        res_df['температура тела'] = res_df['температура тела'].fillna('').astype(str).apply(lambda x: re.sub(pattern_sub,'', x))\
        .apply(lambda x: x.split()[0] if len(x.split())>1 else x).str.strip().apply(temperature_finder)
        
    if 'веc, кг' in res_df.columns:
        res_df['веc, кг'] = res_df['веc, кг'].fillna('').astype(str).apply(lambda x: re.sub(pattern_sub,'', x)).str.strip().apply(weight_height_finder)
        
    if 'рост, см' in res_df.columns:   
        res_df['рост, см'] = res_df['рост, см'].fillna('').astype(str).apply(lambda x: re.sub(pattern_sub,'', x)).str.strip().apply(weight_height_finder)
    
    res_df['ИМТ'] = res_df['веc, кг']/(res_df['рост, см']/100)**2
    
    for c in res_df.select_dtypes('object').columns:
        res_df[c].fillna('не_заполнено', inplace = True)
    
    return pd.concat([res_df, df], axis=1)


def train_val_preprocesses(filenames = ['train.csv','val.csv']):
    for filename in filenames:
        df = file_loader('train.csv').pipe(total_preprocessinag).pipe(sep_columns_df_maker)
        filename_new = 'prepr_' + filename.replace('.csv','')
        df.to_pickle(filename_new)