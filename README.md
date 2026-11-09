# RetailPulse – AI-Powered Retail Analytics & Prediction System

## Project Overview

RetailPulse is an AI-powered retail analytics and prediction system developed using Python, Machine Learning, and Streamlit.

This project was assigned as part of an internship at **Zidio**. It focuses on analyzing retail business data, understanding customer behavior, predicting customer churn, forecasting demand, and providing inventory optimization recommendations through an interactive dashboard.

## Project Features

### 1. Executive Dashboard
- Displays overall retail business performance.
- Presents key performance indicators (KPIs).
- Visualizes important business trends.

### 2. Sales Analytics
- Analyzes total revenue and sales performance.
- Displays monthly revenue trends.
- Provides insights into retail transactions.

### 3. Customer Segmentation
- Performs RFM (Recency, Frequency, Monetary) analysis.
- Applies K-Means clustering.
- Groups customers into meaningful segments.

### 4. Churn Prediction
- Uses a Random Forest classification model.
- Identifies customers based on churn risk.
- Supports customer retention analysis.

### 5. Demand Forecasting
- Uses a Random Forest regression model.
- Forecasts daily retail demand for the next 30 days.
- Visualizes historical and predicted demand.

### 6. Inventory Optimization
- Analyzes product-wise demand.
- Calculates safety stock and reorder points.
- Generates suggested inventory order quantities.
- Identifies exceptional bulk transactions separately.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Data Visualization:** Plotly
- **Web Dashboard:** Streamlit

## Project Structure

```text
RetailPurse/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
├── data/
└── models/
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Eraj05/RetailPulse-AI-Retail-Analytics.git
```

### 2. Navigate to the Project Folder

```bash
cd RetailPulse-AI-Retail-Analytics
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

## Machine Learning Models

| Module | Algorithm | Purpose |
|---|---|---|
| Customer Segmentation | K-Means Clustering | Customer grouping |
| Churn Prediction | Random Forest Classifier | Churn risk prediction |
| Demand Forecasting | Random Forest Regressor | Demand forecasting |
| Inventory Optimization | Statistical Analysis | Inventory recommendations |

## Project Objectives

- Analyze retail sales performance.
- Understand customer purchasing behavior.
- Identify customers at risk of churn.
- Forecast future retail demand.
- Improve inventory planning.
- Present business insights through an interactive dashboard.

## Limitations

- Churn is estimated using customer inactivity as a proxy.
- Demand forecasting uses historical dataset patterns.
- Inventory recommendations are estimates because actual stock-on-hand data is unavailable.
- Two exceptional bulk transactions are excluded from routine inventory calculations as an analytical assumption.

## Future Enhancements

- Real-time retail data integration.
- Product-level demand forecasting.
- Live inventory tracking.
- Advanced machine learning models.
- Cloud deployment.

## Acknowledgement

This project was undertaken as part of an internship at **Zidio**.

## Conclusion

RetailPulse demonstrates the application of data analytics, machine learning, and interactive visualization in retail business operations. It provides insights into sales performance, customer segmentation, churn risk, demand forecasting, and inventory planning.

---

**Project:** RetailPulse – AI-Powered Retail Analytics & Prediction System  
**Internship Organization:** Zidio  
**Domain:** Artificial Intelligence & Data Science
