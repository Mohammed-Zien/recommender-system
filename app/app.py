import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hybrid News Recommender", layout="wide")
st.title("📰 Hybrid News Recommender System")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Recommend", "➕ Add", "📝 Update", "❌ Delete", "📊 Evaluate"])

# --- RECOMMENDATION TAB ---
with tab1:
    st.header("Get News Recommendations")

    news_id = st.text_input("Enter News ID:")
    model = st.selectbox("Model Type", ["bert", "tfidf"])
    alpha = st.slider("Hybrid Score Weight (α)", 0.0, 1.0, 0.5)
    topk = st.number_input("Top-K Recommendations", 1, 50, 10)

    if st.button("Get Recommendations"):
        response = requests.get(f"{API_URL}/get-hybrid-simil/{news_id}", params={
            "model": model,
            "alpha": alpha,
            "topk": topk
        })
        if response.status_code == 200:
            results = response.json()
            st.success("Recommendations received!")
            for item in results:
                st.markdown(f"**{item['News Title']}**")
                st.markdown(f"*{item['Category']} / {item['Subcategory']}*")
                st.write(item['News Abstract'])
                st.write("---")
        else:
            st.error(f"Error: {response.json()['detail']}")

# --- ADD TAB ---
with tab2:
    st.header("Add News Item")
    news_id = st.text_input("News ID", key="add_id")
    category = st.text_input("Category", key="add_cat")
    subcategory = st.text_input("Subcategory", key="add_subcat")
    title = st.text_input("News Title", key="add_title")
    abstract = st.text_area("News Abstract", key="add_abstract")

    if st.button("Add News"):
        if not all([news_id, category, subcategory, title, abstract]):
            st.warning("All fields are required.")
        else:
            payload = {
                "News_ID": news_id,
                "Category": category,
                "Subcategory": subcategory,
                "News_Title": title,
                "News_Abstract": abstract,
}
            r = requests.post(f"{API_URL}/add-news-item", json=payload)
            if r.ok:
                st.success(r.json().get("message", "News deleted!"))
            else:
                st.error(r.json()["detail"])

# --- UPDATE TAB ---
with tab3:
    st.header("Update News Item")
    update_id = st.text_input("Enter News ID to update", key="update_id")

    if st.button("Fetch Current News"):
        if update_id:
            r = requests.get(f"{API_URL}/get-news-by-id/{update_id}")
            if r.ok:
                st.session_state["update_data"] = r.json()
            else:
                st.error(r.json()["detail"])

    if "update_data" in st.session_state:
        news = st.session_state["update_data"]
        category = st.text_input("Category", value=news["Category"], key="upd_cat")
        subcategory = st.text_input("Subcategory", value=news["Subcategory"], key="upd_subcat")
        title = st.text_input("News Title", value=news["News Title"], key="upd_title")
        abstract = st.text_area("News Abstract", value=news["News Abstract"], key="upd_abstract")

        if st.button("Update News"):
            payload = {
                "News_ID": update_id,
                "Category": category,
                "Subcategory": subcategory,
                "News_Title": title,         
                "News_Abstract": abstract    
            }

            r = requests.put(f"{API_URL}/update-news-item/{update_id}", json=payload)
            if r.ok:
                st.success(r.json().get("message", "News deleted!"))
            else:
                st.error(r.json()["detail"])

# --- DELETE TAB ---
with tab4:
    st.header("Delete News Item")
    delete_id = st.text_input("News ID to delete")

    if st.button("Delete"):
        if delete_id:
            r = requests.delete(f"{API_URL}/delete-news-item/{delete_id}")
            if r.ok:
                st.success(r.json().get("message", "News deleted!"))
            else:
                st.error(r.json()["detail"])
        else:
            st.warning("Please enter a News ID to delete.")

# --- EVALUATION TAB ---
with tab5:
    st.header("Evaluate Recommender Model")
    model = st.selectbox("Embedding Model", ["bert", "tfidf"], key="eval_model")
    alpha = st.slider("Hybrid Weight", 0.0, 1.0, 0.5, key="eval_alpha")
    topk = st.slider("Top-K", 1, 50, 10, key="eval_topk")
    n_users = st.slider("Number of Users to Sample", 10, 500, 100)
    seed = st.number_input("Seed (Optional)", value=42)

    if st.button("Evaluate"):
        response = requests.post(f"{API_URL}/evaluate-recommender/", params={
            "model": model,
            "alpha": alpha,
            "topk": topk,
            "n_users": n_users,
            "seed": seed
        })
        if response.status_code == 200:
            metrics = response.json().get("metrics", {})
            st.success("Evaluation complete!")

            st.subheader("📈 Evaluation Metrics")
            for key, value in metrics.items():
                st.markdown(f"**{key}**: {value:.4f}")
        else:
            st.error(f"Error: {response.json()['detail']}")
