import pandas as pd
import torch
import joblib
from scipy.sparse import vstack, save_npz, load_npz
from sentence_transformers import SentenceTransformer
from app import utils

def load_news_database(news_path):
    return utils.load_news_data(news_path)

def save_news_database(df, news_path):
    df.to_csv(news_path, sep="\t", index=True, header=False)

def add_news_item(news_df, news_path, item: dict):
    if item["News_ID"] in news_df.index:
        raise ValueError("News item already exists.")
    
    # Clean and append
    new_row = pd.DataFrame([{
        "Category": item["Category"],
        "Subcategory": item["Subcategory"],
        "News Title": item["News_Title"],  # KEEP
        "News Abstract": item["News_Abstract"],  # KEEP
        "News Url": item.get("News_Url", []),
        "Entities in News Title": item.get("Entities_in_News_Title", []),
        "Entities in News Abstract": item.get("Entities_in_News_Abstract", [])
    }], index=[item["News_ID"]])

    news_df = pd.concat([news_df, new_row], axis=0)
    save_news_database(news_df, news_path)
    return news_df

def update_bert_embedding(news_item, bert_model_path, bert_embedding_path):
    model = SentenceTransformer(bert_model_path)
    content = f'{news_item["Category"]} {news_item["Subcategory"]} {news_item["News_Title"]} {news_item["News_Abstract"]}'
    content = utils.clean_text(content)
    embedding = model.encode(content, convert_to_tensor=True)
    
    existing_embeddings = torch.load(bert_embedding_path)
    updated_embeddings = torch.cat([existing_embeddings, embedding.unsqueeze(0)], dim=0)
    torch.save(updated_embeddings, bert_embedding_path)

def update_tfidf_embedding(news_item, tfidf_model_path, tfidf_embedding_path):
    model = joblib.load(tfidf_model_path)
    content = f'{news_item["Category"]} {news_item["Subcategory"]} {news_item["News_Title"]} {news_item["News_Abstract"]}'
    content = utils.clean_text(content)
    new_embedding = model.transform([content])

    existing_embeddings = load_npz(tfidf_embedding_path)
    updated_embeddings = vstack([existing_embeddings, new_embedding])
    save_npz(tfidf_embedding_path, updated_embeddings)

def delete_news_item(news_id: str, news_path, bert_path, tfidf_path):
    news_df = utils.load_news_data(news_path)

    if news_id not in news_df.index:
        raise ValueError("News ID not found.")

    row_idx = news_df.index.get_loc(news_id)

    # Remove from DataFrame and save
    news_df = news_df.drop(news_id)
    save_news_database(news_df, news_path)

    # Update embeddings
    bert_embeddings = torch.load(bert_path)
    tfidf_embeddings = load_npz(tfidf_path)

    updated_bert = torch.cat([bert_embeddings[:row_idx], bert_embeddings[row_idx+1:]], dim=0)
    updated_tfidf = vstack([tfidf_embeddings[:row_idx], tfidf_embeddings[row_idx+1:]])

    torch.save(updated_bert, bert_path)
    save_npz(tfidf_path, updated_tfidf)

    return news_df

def update_news_item(news_id: str, news_path, item: dict, bert_path, tfidf_path, bert_model_path, tfidf_model_path):
    news_df = utils.load_news_data(news_path)

    if news_id not in news_df.index:
        raise ValueError("News ID not found.")

    row_idx = news_df.index.get_loc(news_id)

    # Update DataFrame
    news_df.loc[news_id] = [
        item["Category"],
        item["Subcategory"],
        item["News_Title"],
        item["News_Abstract"],
        item.get("News_Url", " "),
        item.get("Entities_in_News_Title", []),
        item.get("Entities_in_News_Abstract", [])
    ]

    save_news_database(news_df, news_path)


    # Clean text
    content = f"{item['Category']} {item['Subcategory']} {item['News_Title']} {item['News_Abstract']}"
    content = utils.clean_text(content)

    # Update BERT
    bert_model = SentenceTransformer(bert_model_path)
    new_bert_vec = bert_model.encode(content, convert_to_tensor=True)
    bert_embeddings = torch.load(bert_path)
    bert_embeddings[row_idx] = new_bert_vec
    torch.save(bert_embeddings, bert_path)

    # Update TF-IDF
    tfidf_model = joblib.load(tfidf_model_path)
    new_tfidf_vec = tfidf_model.transform([content])
    tfidf_embeddings = load_npz(tfidf_path)
    tfidf_embeddings = tfidf_embeddings.tolil()
    tfidf_embeddings[row_idx] = new_tfidf_vec
    save_npz(tfidf_path, tfidf_embeddings.tocsr())

    return news_df
