# 📈 Real-Time Stock Price Prediction (Flask App)

This project is a **Real-Time Stock Price Prediction Web Application** built using **Flask**, **Scikit-learn**, and **yfinance**.  
The user enters any stock ticker (Indian or International), the app fetches the latest stock data using **yfinance**, processes it using the same feature engineering pipeline used during training, and predicts the **next day's closing price** using a trained **Linear Regression model**.

---

## 🚀 Features

- 🔍 **Real-time ticker input** from the user  
- 📊 Fetches **live market data** using `yfinance`  
- 🧠 ML model trained on:  
  - Lag features (Lag1, Lag2)  
  - Moving averages (MA5, MA10)  
  - Daily returns  
- 🔮 Predicts **next-day closing price**  
- 🌗 **Dark/Light mode toggle** modern UI  
- ⚡ Fast response with a deployed `.pkl` model  
- 🎨 Clean, professional Flask + HTML/CSS design  

---

## 🏗️ Project Structure

project-folder/
│
├── app/
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── styles.css
│
├── models/
│ └── stock_model.pkl
│
├── notebook/
│ └── model_training.ipynb
│
└── README.md


---

## 📦 Installation & Setup

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
