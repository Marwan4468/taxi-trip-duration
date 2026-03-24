# 🚖 Taxi Trip Duration Prediction using Machine Learning

🔗 Dataset source: [NYC Taxi Trip Duration Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration)

---

## 📌 Overview
This project focuses on predicting taxi trip duration using machine learning techniques. The problem involves capturing complex relationships between **time, distance, and geographic locations**.

---

## 🎯 Objectives
The goal of this project is to:

- Predict trip duration accurately  
- Perform effective feature engineering  
- Compare baseline and advanced models  
- Improve model performance using hyperparameter tuning  

---

## ⚙️ Approach
- Data preprocessing and cleaning  
- Feature engineering:
  - Time-based features  
  - Cyclical encoding (sin/cos)  
  - Distance (Haversine)    
- Model training:
  - Linear Regression (baseline)  
  - XGBoost (tuned using GridSearchCV)  
- Model evaluation using:
  - RMSE  
  - MAE  
  - R² Score  

---

## 📈 Results
- Linear Regression achieved **R² ≈ 0.30**  
- XGBoost significantly improved performance to **R² ≈ 0.77**   

---

## 📊 Key Insights
- Distance is the most important predictor of trip duration  
- Time features (hour/day) affect traffic patterns  
- Geographic coordinates improve model accuracy  
- Advanced models outperform linear models significantly  

---

## 📁 Project Structure

taxi-trip-duration/  
│  
├── main.py    
├── data/  
│   └── NYC.csv  
│  
├── src/  
│   ├── preprocessing.py  
│   ├── feature_engineering.py  
│   ├── train.py  
│   └── evaluate.py  
│  
├── notebooks/  
│   └── eda_and_model.ipynb  
│  
├── requirements.txt  
└── README.md 

---

## ▶️ How to Run

pip install -r requirements.txt  

python main.py  

---

## 🧠 Notes
- Full analysis and visualizations are available in:
  notebooks/eda.ipynb  
- The project follows a modular pipeline structure for scalability  