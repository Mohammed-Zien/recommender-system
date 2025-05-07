
# 📰 Hybrid News Recommender System

A production-ready **hybrid recommender system** for personalized news recommendations. Built using:

- 💡 **Content-Based Filtering**: TF-IDF, BERT, and Doc2Vec
- 👥 **Collaborative Filtering**: Item-based using user behavior logs
- 🔗 **Hybrid Strategy**: Combines multiple methods to improve performance and relevance

### 🚀 Live Demo
You can run the entire application using Docker:
```bash
docker pull mohammedzien/hybrid-news-recommender
docker run -p 8501:8501 mohammedzien/hybrid-news-recommender
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
├── app/
│   ├── main.py              # FastAPI backend
│   ├── app.py               # Streamlit frontend
│   ├── model_assets/        # Downloaded model weights and embeddings
│   ├── recommender.py       # Core hybrid recommendation logic
│   └── utils.py             # Preprocessing, loading, and utility functions
├── requirements.txt
├── start.sh
└── Dockerfile
```

---

## 📦 Features

✅ Personalized recommendations per user  
✅ Evaluate model with precision, recall, nDCG  
✅ Add, update, and delete news entries live  
✅ Fully containerized via Docker

---

## 📊 Models and Data

- **BERT Embeddings** via [SentenceTransformers](https://www.sbert.net/)
- **TF-IDF Embeddings** for text similarity
- **User Interaction Data** modeled via item-item collaborative filtering
- Model assets are hosted on [Hugging Face Hub](https://huggingface.co/MohammedZien/hybrid-news-recommender-assets)

---

## 🐳 Deployment with Docker

1. Clone the repo:
```bash
git clone https://github.com/MohammedZien/hybrid-news-recommender.git
cd hybrid-news-recommender
```

2. Build and run:
```bash
docker build -t hybrid-news-recommender .
docker run -p 8501:8501 hybrid-news-recommender
```

---

## 🤖 Technologies Used

- Python 3.10
- FastAPI
- Streamlit
- PyTorch
- SentenceTransformers
- Scikit-learn
- Docker

---

## 📎 Credits

Built by [Mohammed Zien](https://github.com/MohammedZien) as part of a portfolio-ready data science deployment project.

---

## 🛠️ License

This project is licensed under the MIT License.
