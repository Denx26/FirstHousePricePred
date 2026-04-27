# 🏠 HousePricePred

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

A house price prediction app powered by **XGBoost**, served via **FastAPI** and an interactive **Gradio** chat-like UI.

---

## 📁 Project Structure

HousePricePred/
├── app/
│   ├── api.py           # FastAPI REST API
│   └── gradio_app.py    # Gradio chat UI
├── data/
│   └── Housing.csv      # Dataset
├── models/
│   └── model.joblib     # Trained model
├── train_and_save.py    # Training script
├── Dockerfile
└── requirements.txt

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train & Save Model
```bash
python train_and_save.py data/Housing.csv --out models/model.joblib
```

### 3️⃣ Run the FastAPI
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
POST to `http://localhost:8000/predict`:
```json
{
  "data": {
    "area": 6000, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "mainroad": "yes", "guestroom": "no", "basement": "no",
    "hotwaterheating": "no", "airconditioning": "no", "parking": 1,
    "prefarea": "no", "furnishingstatus": "semi-furnished"
  }
}
```

### 4️⃣ Run Gradio UI
```bash
python app/gradio_app.py
```
Then open `http://localhost:7860` 🎉

### 5️⃣ Run with Docker 🐳
```bash
docker build -t house-price-pred .
docker run -p 7860:7860 house-price-pred
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 1,479,462 |
| R² Score | 0.567 |

---

## 📝 Notes

> ⚠️ Make sure the `scikit-learn` version in `requirements.txt` matches the version used to train the model to avoid version mismatch errors.

- The Gradio UI is intended for **local/quick demos**
- For production, use the **FastAPI** endpoint with a separate frontend
- Deploy to **Railway**, **Fly.io**, or any Docker-compatible host
