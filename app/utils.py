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


def load_tfidf(local_dir="app/model_assets"):
    os.makedirs(local_dir, exist_ok=True)

    emb_path = os.path.join(local_dir, "tfidf_embeddings.npz")
    vec_path = os.path.join(local_dir, "tfidf_vectorizer.pkl")

    if not os.path.exists(emb_path):
        emb_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/tfidf_embeddings.npz",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    if not os.path.exists(vec_path):
        vec_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/tfidf_vectorizer.pkl",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    return load_npz(emb_path), joblib.load(vec_path)

def load_bert(local_dir="app/model_assets"):
    os.makedirs(local_dir, exist_ok=True)

    emb_path = os.path.join(local_dir, "bert_embeddings.pt")
    model_path = os.path.join(local_dir, "bert_model")

    if not os.path.exists(emb_path):
        emb_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/bert_embeddings.pt",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    if not os.path.exists(model_path):
        model_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/bert_model/config.json",  # any file inside the model dir
            local_dir=os.path.join(local_dir, "bert_model"),
            local_dir_use_symlinks=False
        )
        model_path = os.path.dirname(model_path)

    return torch.load(emb_path), SentenceTransformer(model_path)

def load_item_sim_df(local_dir="app/model_assets"):
    file_path = os.path.join(local_dir, "item_sim_df.pkl")
    if not os.path.exists(file_path):
        file_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/item_sim_df.pkl",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    return pd.read_pickle(file_path)

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

def load_news_data(local_dir="app/model_assets"):
    file_path = os.path.join(local_dir, "news.tsv")
    if not os.path.exists(file_path):
        file_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/news.tsv",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    return pd.read_csv(file_path,
                       names=["News ID", "Category", "Subcategory", "News Title", "News Abstract", "News Url", "Entities in News Title", "Entities in News Abstract"],
                       index_col="News ID", delimiter="\t").fillna("").sort_index()

def load_behaviors_data(local_dir="app/model_assets"):
    file_path = os.path.join(local_dir, "behaviors.tsv")
    if not os.path.exists(file_path):
        file_path = hf_hub_download(
            repo_id="MohammedZien/hybrid-news-recommender-assets",
            filename="model_assets/behaviors.tsv",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    return pd.read_csv(file_path,
                       names=["Impression ID", "User ID", "Impression Time", "User Click History", "Impression News"],
                       delimiter="\t")

def download_all_assets(local_dir="app/model_assets"):
    os.makedirs(local_dir, exist_ok=True)

    # Load each component
    print("Downloading BERT model & embeddings...")
    bert_embeddings, bert_model = load_bert(local_dir)

    print("Downloading TF-IDF model & embeddings...")
    tfidf_embeddings, tfidf_model = load_tfidf(local_dir)

    print("Downloading Item Similarity Matrix...")
    item_sim_df = load_item_sim_df(local_dir)

    print("Downloading News Dataset...")
    news = load_news_data(local_dir)

    print("Downloading User Behavior Data...")
    behaviors = load_behaviors_data(local_dir)

    return {
        "bert_embeddings": bert_embeddings,
        "bert_model": bert_model,
        "tfidf_embeddings": tfidf_embeddings,
        "tfidf_model": tfidf_model,
        "item_sim_df": item_sim_df,
        "news": news,
        "behaviors": behaviors
    }
