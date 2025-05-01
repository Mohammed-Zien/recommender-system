from ctypes import util
import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers.util import cos_sim
import torch

def tfidf_recommendation(news, target, embedding_bank, model, topk=10):
    content = target["Category"] + " " + target["Subcategory"] + " " + target["News Title"] + " " + target["News Abstract"]
    target = utils.clean_text(content)
    target_vector = model.transform([target])

    similarities = cosine_similarity(embedding_bank, target_vector).flatten()
    topk_indices = similarities.argsort()[::-1][1:topk+1]

    top_simil = [
                    {
                        "News ID": news.index[i],
                        "Category": news.iloc[i]["Category"],
                        "Subcategory": news.iloc[i]["Subcategory"],
                        "News Title": news.iloc[i]["News Title"],
                        "News Abstract": news.iloc[i]["News Abstract"],
                        "Similarity": round(float(similarities[i]), 4)
                    }
                   for i in topk_indices]

    return top_simil

def bert_recommendation(news, target, embedding_bank, model, topk=10):
    content = target["Category"] + " " + target["Subcategory"] + " " + target["News Title"] + " " + target["News Abstract"]
    target_vector = model.encode(content, convert_to_tensor=True)

    similarities = cos_sim(target_vector, embedding_bank)[0].cpu().numpy()

    topk_indices = similarities.argsort()[::-1][1:topk+1]  # +1 to skip identical article

    # Step 5: Build response
    top_simil = [
                    {
                        "News ID": news.index[i],
                        "Category": news.iloc[i]["Category"],
                        "Subcategory": news.iloc[i]["Subcategory"],
                        "News Title": news.iloc[i]["News Title"],
                        "News Abstract": news.iloc[i]["News Abstract"],
                        "Similarity": round(float(similarities[i]), 4)
                    }
                   for i in topk_indices]

    return top_simil
class HybridRecommender:
    def __init__(self, news, target, embeddings, model, item_sim_df,
                 mode="bert", alpha=0.5, topk=10):
        """
        mode: "bert" or "tfidf"
        """
        self.data = news  # full sorted news data (indexed)
        self.target = target  # row of the target article
        self.embeddings = embeddings
        self.model = model
        self.item_sim_df = item_sim_df  # sparse DF
        self.alpha = alpha
        self.topk = topk
        self.mode = mode

    def recommend(self, target_id):
        content_score = self.calculate_content_score()
        cf_score = self.calculate_cf_score(target_id)
        return self.combine_scores(content_score, cf_score)

    def calculate_content_score(self):
        content = (
            self.target["Category"] + " " +
            self.target["Subcategory"] + " " +
            self.target["News Title"] + " " +
            self.target["News Abstract"]
        )

        if self.mode == "bert":
            target_vector = self.model.encode(content, convert_to_tensor=True)
            similarities = cos_sim(target_vector, self.embeddings)[0].cpu().numpy()

        elif self.mode == "tfidf":
            clean_text = utils.clean_text(content)
            target_vector = self.model.transform([clean_text])
            similarities = cosine_similarity(self.embeddings, target_vector).flatten()

        else:
            raise ValueError("Invalid mode. Choose 'bert' or 'tfidf'.")

        return similarities

    def calculate_cf_score(self, target_id):
        # Ensure it's in index
        if target_id not in self.item_sim_df.columns:
            return pd.Series([0] * len(self.data), index=self.data.index)

        # Sparse matrix column → full Series (pandas handles sparse+dense fine)
        cf_series = self.item_sim_df[target_id]
        return cf_series.to_numpy()


    def normalize(self, score):
        # score can be Series or ndarray
        if isinstance(score, pd.Series):
            values = score.values
        else:
            values = score

        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val < 1e-8:
            return pd.Series([0.0] * len(values), index=self.data.index)

        norm = (values - min_val) / (max_val - min_val + 1e-8)
        return np.round(norm, 2)

    def combine_scores(self, content_score, cf_score):
        content_score = np.round(self.normalize(content_score), 4)
        cf_score = pd.Series(np.round(self.normalize(cf_score), 4), index=self.item_sim_df.index, name="cf_score")

        # Attach scores to dataframe
        result_df = self.data[["Category", "Subcategory", "News Title", "News Abstract"]].copy()
        result_df["content_score"] = content_score
        result_df = pd.merge(result_df, cf_score, left_on="News ID", right_on=cf_score.index,how="left")
        result_df["cf_score"] = result_df["cf_score"].fillna(0)
        result_df["hybrid_score"] = np.round(self.alpha * result_df["content_score"] + (1 - self.alpha) * result_df["cf_score"], 4)

        return result_df.sort_values("hybrid_score", ascending=False).head(self.topk).to_dict(orient="records")

