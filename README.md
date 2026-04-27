Project: FirstHousePricePred

Quick start

1) Install dependencies (prefer a virtualenv):

```bash
pip install -r requirements.txt
```

2) Train and save model (saves to `models/model.joblib`):

```bash
python train_and_save.py data/Housing.csv --out models/model.joblib
```

3) Run the API (FastAPI):

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

POST JSON to `http://localhost:8000/predict` with body like:

```json
{ "data": { "area": 6000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "no", "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished" } }
```

4) Run Gradio chat-like UI (opens local web UI):

```bash
python app/gradio_app.py
```

5) Run with Docker:
```bash
# Build the image
docker build -t house-price-pred .

# Run the container
docker run -p 7860:7860 house-price-pred
```

Notes
- The Gradio UI provides a simple chat-like interface for providing structured features and seeing the model reply.
- For deployment, you can deploy the FastAPI app to services such as Railway, Fly.io, or a Docker container on any host. The Gradio app is intended for local/quick demos; for production use the API and a separate frontend.

