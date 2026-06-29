# Enterprise Sales Analytics Platform 🚀

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://imran17-wq-enterprise-sales-analytics-dashboard-w8f2uq.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)

**An enterprise-grade, highly interactive Sales Intelligence Platform built entirely in Python.**

</div>

---

## 📸 Dashboard Preview

![Dashboard Preview](https://via.placeholder.com/1200x600.png?text=Executive+Command+Center+Preview)

---

## 🌟 Key Highlights

✅ **Interactive Sales Analytics Dashboard**  
✅ **Machine Learning Forecasting**  
✅ **KPI Monitoring & Automated Evaluation**  
✅ **Advanced Business Intelligence**  
✅ **Dynamic Filtering & Drill-Down Breadcrumbs**  
✅ **Streamlit Cloud Deployment Ready**  
✅ **Portfolio-Ready Project Architecture**  

---

## 📊 Project Metrics

- **📊 Interactive Visualizations**: Dynamically routed visualizations based on dimensionality.
- **📈 Forecasting Models**: Exponential Smoothing & Linear Trendlines predicting future sales.
- **🧠 Business Intelligence Features**: AI-powered textual summaries & variance analysis.
- **🔍 Advanced Analytics**: Isolation Forest anomaly detection for identifying outliers.
- **☁️ Cloud Deployment**: Fully containerized and deployed on Streamlit Community Cloud.

---

## 📖 Overview

The **Enterprise Sales Analytics Platform** transforms raw sales data into actionable executive intelligence. It was engineered to mimic the capabilities of tier-1 BI tools (Tableau, Power BI, Looker) while remaining purely Pythonic. 

By leveraging **Pandas** for high-speed aggregations, **Plotly** for interactive visualizations, and **Scikit-Learn** for predictive models, this platform bridges the gap between traditional Business Intelligence and modern Data Science. It features a luxury SaaS dark-mode aesthetic built completely from custom CSS injections.

---

## ✨ Core Features

### 🔭 Interactive Business Explorer
A unified module allowing dynamic dimension and metric selection to auto-generate the optimal visualizations (Sunbursts, Donuts, Grouped Bars). 
- **Drill-Down Breadcrumbs**: Click on charts to drill into the underlying data contextually.
- **Dynamic Pivot Tables**: Generate Pandas pivot tables dynamically based on dropdown selections.

### ⚖️ Advanced Compare Mode
Select two entities (e.g., North vs South, Tablet vs Laptop) to generate side-by-side KPIs, performance deltas, and automated **AI Insights**.

### 🧠 Executive Command Center & AI Summaries
- **Health & Growth Scores**: Proprietary algorithmic evaluation of business health.
- **Automated Textual Summaries**: Generates a breakdown explaining the "Why" behind the data, identifying key drivers and underperforming segments automatically.

### ⚠️ Machine Learning Anomaly Detection
Uses an **Isolation Forest** (Scikit-Learn) to identify statistically significant outliers in transactional data to prevent fraud or flag extraordinary sales events.

### 🎛️ Business Scenario Simulator (What-If)
Adjust business levers (e.g., Discount Rate, Demand Volume) using sliders to project instant revenue impacts via multiplicative elasticity models.

---

## 📸 Feature Previews

### Main Dashboard
![Main Dashboard](https://via.placeholder.com/1000x500.png?text=Main+Dashboard+KPIs)

### Forecasting Section
![Forecasting Section](https://via.placeholder.com/1000x500.png?text=Machine+Learning+Forecasting)

### Business Intelligence Section
![Business Intelligence Section](https://via.placeholder.com/1000x500.png?text=Executive+Summary+%26+Anomalies)

### Interactive Analysis Section
![Interactive Analysis Section](https://via.placeholder.com/1000x500.png?text=Business+Explorer+%26+Pivot+Tables)

*(Note: Replace placeholder images above with actual screenshots of the application before publishing).*

---

## 🏗️ Technical Architecture & Project Structure

The project has been modularized for scalability and maintainability:

```text
enterprise-sales-analytics/
│
├── dashboard.py               # Main Streamlit Entry Point
├── requirements.txt           # Cloud-ready dependencies
├── README.md                  # Project Documentation
├── .gitignore                 # Git ignore file
├── .streamlit/                # Cloud Configs
│   └── config.toml            # Enforces dark mode theme
│
├── data/
│   └── Product-Sales-Region.xlsx  # Dataset (dynamically loaded)
│
└── modules/                   # Core Python logic
    ├── __init__.py
    ├── data_loader.py         # OS pathing and Excel ingestion
    ├── data_cleaning.py       # Data sanitization pipeline
    ├── feature_engineering.py # Enrichment (RFM, margins, etc.)
    ├── forecasting.py         # Predictive modeling functions
    ├── intelligence.py        # Executive Summaries, Anomalies, What-If
    ├── explorer.py            # Tableau-style Auto-Charting & Drill-Down
    └── utils.py               # Theme CSS engine and KPI logic
```

---

## 💻 Installation Guide (Local Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/imran17-wq/enterprise-sales-analytics.git
   cd enterprise-sales-analytics
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run dashboard.py
   ```

5. **Access the Dashboard**:
   Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

This project has been fully audited and prepared for zero-configuration deployment on **Streamlit Community Cloud**.

1. **Push your code to GitHub** (If you haven't already).
2. **Log into Streamlit Community Cloud**: Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. **Deploy a New App**:
   - Click **New app**.
   - Select your repository: `imran17-wq/enterprise-sales-analytics`.
   - Keep the branch as `main`.
   - Set the **Main file path** to `dashboard.py`.
4. **Deploy**: Click the **Deploy!** button. 

Streamlit will automatically read `requirements.txt`, install dependencies, apply the `.streamlit/config.toml` dark mode settings, and launch your dashboard globally!

---

## 🚀 Future Roadmap

- [ ] **SQL Database Integration**: Migrate from Excel to a Postgres/Snowflake data warehouse.
- [ ] **Real-Time Data Streaming**: Implement Kafka pipelines for live updates.
- [ ] **LLM Integration**: Connect OpenAI API for dynamic natural language querying (e.g., "Why did revenue drop in Q3?").
- [ ] **Advanced Forecasting**: Implement Prophet or Deep Learning models for time-series predictions.
- [ ] **Authentication**: Add JWT-based user login and role-based access control (RBAC).

---

<div align="center">

*Built with 💻 and ☕ to transform data into decisions.*

</div>
