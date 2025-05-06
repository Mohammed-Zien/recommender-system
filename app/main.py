import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Literal
from pathlib import Path
import random
from app import utils, recommender, news_database, Evaluator

# Base directory setup
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "model_assets"

print("ASSETS_DIR =", ASSETS_DIR)
print("Files:", os.listdir(ASSETS_DIR))

app = FastAPI()

# Load core data
print("LOADING ASSETS")
behaviors = utils.load_behaviors_data(str(ASSETS_DIR / "behaviors.tsv"))
item_sim_df = utils.load_item_sim_df(str(ASSETS_DIR / "item_sim_df.pkl"))
user_clicks = utils.get_user_clicks(behaviors)


# ========== Data Models ========== #
class NewsItem(BaseModel):
    News_ID: str
    Category: str
    Subcategory: str
    News_Title: str
    News_Abstract: str
    News_Url: Optional[str] = " "
    Entities_in_News_Title: Optional[list] = []
    Entities_in_News_Abstract: Optional[list] = []


# ========== Endpoints ========== #

@app.get("/get-news-by-id/{News_ID}")
def get_news_by_id(News_ID: str):
    news = utils.load_news_data(str(ASSETS_DIR / "news.tsv"))
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")
    return news.loc[News_ID].to_dict()

@app.get("/get-hybrid-simil/{News_ID}")
def get_hybrid_simil(
    News_ID: str,
    topk: int = 10,
    model: Literal["bert", "tfidf"] = "bert",
    alpha: float = 0.5
):
    news = utils.load_news_data(str(ASSETS_DIR / "news.tsv"))
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")

    if model == "bert":
        embeddings, embed_model = utils.load_bert(str(ASSETS_DIR / "bert_embeddings.pt"), str(ASSETS_DIR / "bert_model"))
    else:
        embeddings, embed_model = utils.load_tfidf(str(ASSETS_DIR / "tfidf_embeddings.npz"), str(ASSETS_DIR / "tfidf_vectorizer.pkl"))

    recommender_obj = recommender.HybridRecommender(
        news=news,
        embeddings=embeddings,
        model=embed_model,
        item_sim_df=item_sim_df,
        mode=model,
        alpha=alpha,
        topk=topk
    )
    return recommender_obj.recommend(News_ID)

@app.post("/add-news-item")
def add_news(item: NewsItem):
    try:
        news_df = utils.load_news_data(str(ASSETS_DIR / "news.tsv"))
        news_database.add_news_item(news_df, str(ASSETS_DIR / "news.tsv"), item.model_dump())

        news_database.update_bert_embedding(item.model_dump(), str(ASSETS_DIR / "bert_model"), str(ASSETS_DIR / "bert_embeddings.pt"))
        news_database.update_tfidf_embedding(item.model_dump(), str(ASSETS_DIR / "tfidf_vectorizer.pkl"), str(ASSETS_DIR / "tfidf_embeddings.npz"))

        return {"status": "success", "message": f'News "{item.News_ID}" added and encoded.'}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.put("/update-news-item/{news_id}")
def update_news_item_endpoint(news_id: str, item: NewsItem):
    try:
        news_database.update_news_item(
            news_id,
            str(ASSETS_DIR / "news.tsv"),
            item.model_dump(),
            str(ASSETS_DIR / "bert_embeddings.pt"),
            str(ASSETS_DIR / "tfidf_embeddings.npz"),
            str(ASSETS_DIR / "bert_model"),
            str(ASSETS_DIR / "tfidf_vectorizer.pkl")
        )
        return {"status": "success", "message": f"{news_id} updated."}

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-news-item/{news_id}")
def delete_news_item_endpoint(news_id: str):
    try:
        news_database.delete_news_item(
            news_id,
            str(ASSETS_DIR / "news.tsv"),
            str(ASSETS_DIR / "bert_embeddings.pt"),
            str(ASSETS_DIR / "tfidf_embeddings.npz")
        )
        return {"status": "success", "message": f"{news_id} deleted."}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-recommender/")
def evaluate_recommender(
    model: Literal["bert", "tfidf"] = Query("bert"),
    alpha: float = Query(0.5, ge=0, le=1),
    topk: int = Query(10, gt=0),
    n_users: int = Query(100, gt=0),
    seed: Optional[int] = None
):
    news_df = utils.load_news_data(str(ASSETS_DIR / "news.tsv"))
    user_click_sample = utils.get_user_clicks(behaviors)

    if model == "bert":
        embeddings, embed_model = utils.load_bert(str(ASSETS_DIR / "bert_embeddings.pt"), str(ASSETS_DIR / "bert_model"))
    else:
        embeddings, embed_model = utils.load_tfidf(str(ASSETS_DIR / "tfidf_embeddings.npz"), str(ASSETS_DIR / "tfidf_vectorizer.pkl"))

    user_ids = list(user_click_sample.keys())
    if seed:
        random.seed(seed)
    sampled_users = random.sample(user_ids, min(n_users, len(user_ids)))
    sampled_clicks = {u: user_click_sample[u] for u in sampled_users}

    recommender_obj = recommender.HybridRecommender(
        news=news_df,
        embeddings=embeddings,
        model=embed_model,
        item_sim_df=item_sim_df,
        mode=model,
        alpha=alpha,
        topk=topk
    )

    evaluator = Evaluator.Evaluator(recommender_obj, sampled_clicks, k=topk)
    results = evaluator.evaluate_all()

    return {
        "model": model,
        "alpha": alpha,
        "topk": topk,
        "n_users": len(sampled_users),
        "metrics": results
    }