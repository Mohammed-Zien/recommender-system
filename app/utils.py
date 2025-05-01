import os
import re
import string
import joblib
import torch
import pandas as pd
from scipy.sparse import load_npz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


def load_tfidf(emb_path, model_path):
    return load_npz(emb_path), joblib.load(model_path)

def load_bert(emb_path, model_path):
    return torch.load(emb_path), SentenceTransformer(model_path)

def load_item_sim_df(path):
    return pd.read_pickle(path)

def clean_text(text):
    stopwords_english = stopwords.words('english')
    stemmer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    text = ' '.join(word for word in text.split(' ') if word not in stopwords_english)
    text = ' '.join(stemmer.lemmatize(word) for word in text.split(" "))

    return text

def load_news_data(path):
    return pd.read_csv(path,
                       names=["News ID", "Category", "Subcategory", "News Title", "News Abstract", "News Url", "Entities in News Title", "Entities in News Abstract"],
                       index_col="News ID", delimiter="\t").fillna("").sort_index()

def load_behaviors_data(path):
    return pd.read_csv(path,
                       names=["Impression ID", "User ID", "Impression Time", "User Click History", "Impression News"],
                       delimiter="\t")

def get_user_clicks(users):
    return users[["User ID", "User Click History"]].fillna("").groupby("User ID")["User Click History"].apply(lambda list : sum(list.str.split(), [])).to_dict()