import pandas as pd
import time
import streamlit as st
import plotly.express as px
import pickle

@st.cache
def load_dataset(data_link):
    dataset = pd.read_pickle(data_link)
    return dataset

@st.cache(allow_output_mutation=True)
def load_model(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model


my_data = load_dataset('X_val_sample')
model = load_model('predict_model')

#st.dataframe(my_data)
#st.title('My First Streamlit App')

selected_id = st.selectbox("Выберите id пациента для отображения рисковой информации", my_data['id'].unique())
filtered_df = my_data[my_data['id'] == selected_id]
#st.dataframe(filtered_df, 20000, 100)


def result_df_func(df, client_id):
    uncklean_list = df.loc[client_id,'text_split']
    clean_list = df.loc[client_id,'text_split_cleaned']
    res = []
    for ul, cl in zip(uncklean_list, clean_list):
        if df.loc[client_id,'predict_proba'] > 0.2:
            if len(cl) > 50:
                pred_proba = model.predict_proba([cl])[:,1][0]
                if pred_proba > 0.4:
                    res.append([round(pred_proba,1), ul])
    if len(res)>0:
        res = pd.DataFrame(res).sort_values(by=0, ascending = False).drop_duplicates()
        res.columns = ['Риск','Описание                                                       ']
        return res
    else:
        return pd.DataFrame()
    


new_df = result_df_func(my_data, selected_id)
if new_df.shape[0]>0:
    st.dataframe(new_df.style.background_gradient(axis=1, vmin=0.4, vmax=0.7,cmap='YlOrRd'), 200000, 1000)
else:
    st.title('По пациенту высокорисковых записей не обнаружено')
