# UnderScore - Football Predictor

A streamlined football prediction platform using a custom XGBoost AI pipeline and a lightweight HTML/CSS/JS frontend.

## 🚀 Features
- **FastAPI Backend**: Robust API for training, individual predictions, SHAP insights, and batch processing.
- **Lightweight Frontend**: Fast and clean HTML/CSS/JS dashboard (SPA) served via Python's HTTP server.
- **CSV Data Source**: Uses local CSV datasets for training and inference (PSL 2024/25 & 2025/26).

## 🛠 Running the Application

Simply use the provided start script:
```bash
./start.sh
```
This will launch:
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)

## 📁 Project Structure
- `app/`: Frontend files (HTML, CSS, JS).
- `backend/`: FastAPI application and ML pipeline.
- `backend/data/`: Local CSV datasets.
- `backend/models/`: Cached ML models and statistics.

## 📡 Endpoints overview
- `POST /predict`: Individual match predictions.
- `POST /train`: Retrain the XGBoost model with Optuna optimization.
- `GET /explain`: SHAP feature importance data.
- `POST /batch`: Batch prediction from CSV.

Enjoy the platform!
