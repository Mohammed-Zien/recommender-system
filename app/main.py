from fastapi import HTTPException
from typing import Literal, Optional
from fastapi import FastAPI, Path
import pandas as pd
from pydantic import BaseModel
import utils
import recommender
import news_database

app = FastAPI()

print("LOADING ASSETS")
behaviors = utils.load_behaviors_data("F:\projects\Porfolio\\recommender-system\\app\model_assets\\behaviors.tsv")
item_sim_df = utils.load_item_sim_df("F:\\projects\\Porfolio\\recommender-system\\app\model_assets\\item_sim_df.pkl")
user_clicks = utils.get_user_clicks(behaviors)

class NewsItem(BaseModel):
    News_ID: str
    Category: str
    Subcategory: str
    News_Title: str
    News_Abstract: str
    News_Url: Optional[str] = " "
    Entities_in_News_Title: Optional[list] =[]
    Entities_in_News_Abstract: Optional[list] = []

@app.get("/get-news-by-id/{News_ID}")  
def get_news_by_id(News_ID: str):
    news = utils.load_news_data("F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv")
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
    news = utils.load_news_data("F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv")
    if News_ID not in news.index:
        raise HTTPException(status_code=404, detail="News ID not found")

    target_row = news.loc[News_ID]

    if model == "bert":
        bert_embeddings, bert_model= utils.load_bert("F:\projects\Porfolio\\recommender-system\\app\model_assets\\bert_embeddiings.pt",
                                   "F:\projects\Porfolio\\recommender-system\\app\model_assets\\bert_model")
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
        tfidf_embeddings, tfidf_model = utils.load_tfidf("F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_embeddings.npz",
                                     "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_vectorizer.pkl")
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

@app.post("/add-news-item")
def add_news(item: NewsItem):
    try:
        news_df = utils.load_news_data("F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv")
        news_database.add_news_item(news_df, "F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv", item.model_dump())

        news_database.update_bert_embedding(item.model_dump(), "F:/projects/Porfolio/recommender-system/app/model_assets/bert_model",
                                                         "F:\projects\Porfolio\\recommender-system\\app\model_assets\\bert_embeddiings.pt")
        
        news_database.update_tfidf_embedding(item.model_dump(), "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_vectorizer.pkl",
                                                          "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_embeddings.npz")

        return {"status": "success", "message": f'News "{item.News_ID}" added and encoded.'}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.delete("/delete-news-item/{news_id}")
def delete_news_item_endpoint(news_id: str):
    try:
        news_database.delete_news_item(news_id,
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv",
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\bert_embeddiings.pt",
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_embeddings.npz"
        )
        return {"status": "success", "message": f"{news_id} deleted."}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.put("/update-news-item/{news_id}")
def update_news_item_endpoint(news_id: str, item: NewsItem):
    try:
        news_database.update_news_item(news_id,
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\news.tsv",
            item.model_dump(),
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\bert_embeddiings.pt",
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_embeddings.npz",
            "F:/projects/Porfolio/recommender-system/app/model_assets/bert_model",
            "F:\projects\Porfolio\\recommender-system\\app\model_assets\\tfidf_vectorizer.pkl"
        )
        return {"status": "success", "message": f"{news_id} updated."}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
