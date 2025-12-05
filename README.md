# House_predictor
End-to-end machine learning regression system that predicts California housing prices using a PyTorch neural network. Includes data preprocessing, scaling, model training with early stopping, evaluation, FastAPI inference server, and Streamlit UI for real-time prediction.

# 🏠 California Housing Price Prediction (PyTorch + FastAPI + Streamlit)

This project builds a **Machine Learning regression model** using **PyTorch**, trained on the **California Housing Dataset**, and exposes the model via a **FastAPI web backend** with a **Streamlit UI frontend** for user interaction.

It represents a complete ML workflow:

- Real dataset → Preprocessing → Neural Network Training → Evaluation → Model Artifact Saving → API Deployment → UI Interface.


---

## 📌 Features

✔ Trainable PyTorch neural network  
✔ Proper train/validation/test split  
✔ Feature scaling (StandardScaler)  
✔ Early stopping for best generalization  
✔ Performance metrics (RMSE, MAE, R²)  
✔ FastAPI REST API for inference  
✔ Streamlit interactive UI  
✔ GPU supported (CUDA enabled)  
✔ Saved artifacts (`model_best.pth`, `scaler.joblib`)  
✔ End-to-end project ready for deployment  

---

## 🚀 Tech Stack

 Component         |   Technology 

Training Framework |   PyTorch  
Data Processing    |   Pandas, NumPy, Sklearn  
API Backend        |   FastAPI  
Frontend UI        |   Streamlit  
Model Saving       |   Torch state_dict + Scaler (joblib)  
Visualization      |   Matplotlib  
Deployment-ready   |   Uvicorn / Docker (optional)  

---

## 📂 Project Structure

📦 California-Housing-Price-Prediction
├── src/
│ ├── data.py # Dataset loading, splitting, scaling
│ ├── train.py # Training + validation + test
│ ├── evaluate.py # (Optional) separate evaluation
│ ├── api.py # FastAPI model serving
├── ui_app.py # Streamlit frontend
├── data/
│ ├── scaler.joblib # Saved scaler from preprocessing
│ └── dataset_scaled.npz # Preprocessed dataset splits
├── artifacts/
│ ├── model_best.pth # Best trained model
│ ├── model_final.pth # Last epoch model
│ └── test_metrics.json # Final metrics (RMSE, MAE, R²)
└── requirements.txt


---

## 📊 Dataset Information

**Dataset:** California Housing Dataset  

The dataset contains features such as:

- Median Income  
- House Age  
- Average Rooms  
- Average Bedrooms  
- Population  
- Average Occupancy  
- Latitude  
- Longitude  

**Target (y):**
> Median house value (in units of 100,000 USD)

Dataset source:
- `sklearn.datasets.fetch_california_housing()`


---

## 🧠 Model Architecture (Baseline)

python
( Input: 8 features )
 → Linear(8 → 64)
 → ReLU
 → Linear(64 → 32)
 → ReLU
 → Linear(32 → 1)
( Output: Predicted house price )
Chosen because:

Light and efficient

Avoids overfitting

Provides strong generalization

📈 Model Performance
Metric	Value
RMSE	~0.4969
MAE	~0.3401
R² Score	~0.8133

This means:

~81% of price variance explained

Good for tabular regression ML

🏋️ Training Procedure
Train/Validation/Test = 70% / 15% / 15%

Adam optimizer (LR = 0.001)

Early stopping to avoid overfitting

Best model saved as model_best.pth

Scaling using StandardScaler

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate  # Linux/macOS
3. Install dependencies
pip install -r requirements.txt

🧪 Training the Model
Run:
python src/train.py --out_dir ./artifacts --data_dir ./data

Artifacts created:
Trained model → model_best.pth

Scaler → scaler.joblib

Metrics → test_metrics.json

Loss Curve Plot → loss_plot.png

🔍 Evaluate the Model (Optional)
python src/evaluate.py

📡 Running the FastAPI Backend
uvicorn src.api:app --reload
Server runs at:
http://127.0.0.1:8000
Swagger API docs at:
arduino
http://127.0.0.1:8000/docs

🎨 Running the Streamlit UI
Open a new terminal (while FastAPI is running):
streamlit run ui_app.py

Streamlit runs at:
arduino
http://localhost:8501

🧭 API Example Request
json
POST http://127.0.0.1:8000/predict
{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 800.0,
  "AveOccup": 3.0,
  "Latitude": 34.2,
  "Longitude": -118.5
}
Response:
json
{
  "predicted_value_100k": 2.66079,
  "predicted_value_dollars": 266079.0
}

🚀 Deployment Options
Railway (recommended for free)

Render

Streamlit Cloud + external API

Docker + Cloud Hosting

📘 Future Improvements
Add batch prediction

Add confidence intervals

Train boosted tree model (XGBoost)

Hyperparameter tuning

Deploy on Render/Railway with Docker

📝 License
This project is released under the MIT license.

👨‍💻 Author
Jagrat Jani
Python | ML | AI | Backend

⭐ If you like this project…
Please ⭐ the repository to support future development!
