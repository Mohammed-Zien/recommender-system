from http.client import HTTPException
from typing import Optional
from fastapi import FastAPI, Path
import pandas as pd
from pydantic import BaseModel
import utils
import recommender

app = FastAPI()

assets = utils.download_all_assets()

# Access components
tfidf_embeddings = assets["tfidf_embeddings"]
tfidf_model = assets["tfidf_model"]
bert_embeddings = assets["bert_embeddings"]
bert_model = assets["bert_model"]
news = assets["news"]
behaviors = assets["behaviors"]
item_sim_df = assets["item_sim_df"]

class NewsItem(BaseModel):
    Category: str
    Subcategory: str
    News_Title: str
    News_Abstract: str
    News_Url: Optional[str] = None
    Entities_in_News_Title: Optional[str] = None
    Entities_in_News_Abstract: Optional[str] = None


@app.get("/get-news-by-id/{News_ID}")  
def get_news_by_id(News_ID : str):
    if News_ID not in news.index :
        raise HTTPException(status_code=404, detail="News ID not found")
    
    return news.loc[News_ID].to_dict()

@app.get("/get-tfidf-simil/{News_ID}")
def get_tfidf_simil(News_ID, topk:int=10):
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")
    
    return recommender.tfidf_recommendation(news, news.loc[News_ID], tfidf_embeddings, tfidf_model, topk)

@app.get("/get-bert-simil/{News_ID}")
def get_bert_simil(News_ID, topk:int=10):
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")
    
    return recommender.bert_recommendation(news, news.loc[News_ID], bert_embeddings, bert_model, topk)

from fastapi import HTTPException
from typing import Literal

@app.get("/get-hybrid-simil/{News_ID}")
def get_hybrid_simil(
    News_ID: str,
    topk: int = 10,
    model: Literal["bert", "tfidf"] = "bert",
    alpha: float = 0.5
):
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")

    target_row = news.loc[News_ID]

    if model == "bert":
        recommender_ = recommender.HybridRecommender(
            news=news,
            target=target_row,
            embeddings=bert_embeddings,
            model=bert_model,
            item_sim_df=item_sim_df,
            mode="bert",
            alpha=alpha,
            topk=topk
        )
    elif model == "tfidf":
        recommender_ = recommender.HybridRecommender(
            news=news,
            target=target_row,
            embeddings=tfidf_embeddings,
            model=tfidf_model,
            item_sim_df=item_sim_df,
            mode="tfidf",
            alpha=alpha,
            topk=topk
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'bert' or 'tfidf'.")

    return recommender_.recommend(News_ID)

    



