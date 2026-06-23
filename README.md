# Enterprise Sales Analytics Platform 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, highly interactive **Sales Intelligence Platform** built entirely in Python using Streamlit. Designed with a luxury SaaS dark-mode aesthetic, this dashboard moves beyond static reporting to provide dynamic exploration, advanced analytics, anomaly detection, and predictive forecasting.

![Dashboard Preview](https://via.placeholder.com/1200x600.png?text=Executive+Command+Center)

---

## 📖 Overview

The **Enterprise Sales Analytics Platform** transforms raw sales data into actionable executive intelligence. It was engineered to mimic the capabilities of tier-1 BI tools (Tableau, Power BI, Looker) while remaining purely Pythonic. 

By leveraging **Pandas** for high-speed aggregations, **Plotly** for interactive visualizations, and **Scikit-Learn** for predictive models, this platform bridges the gap between traditional BI and modern Data Science.

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

### 🔮 Statistical Forecasting
Predicts future sales trajectories using exponential smoothing and linear regression trendlines, factoring in historical seasonality.

### 🎨 Premium SaaS Aesthetic
Frosted glassmorphism (`backdrop-filter`), responsive hover micro-interactions, rich indigo/navy gradients, and an exclusive deep dark theme built entirely via custom CSS injection.

---

## 🛠️ Tech Stack

- **Frontend / Framework**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Plotly Express & Graph Objects
- **Machine Learning / Stats**: Scikit-Learn (Isolation Forest), SciPy
- **Styling**: Custom CSS (Inter Font, Flexbox, Glassmorphism)

---

## 🏗️ Architecture & Project Structure

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

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
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
   - Keep the branch as `master`.
   - Set the **Main file path** to `dashboard.py`.
4. **Deploy**: Click the **Deploy!** button. 

Streamlit will automatically read `requirements.txt`, install dependencies, apply the `.streamlit/config.toml` dark mode settings, and launch your dashboard globally!

---

## 👨‍💻 Author

Built as a portfolio demonstration of advanced Data Science, Business Intelligence, and UX Design.

*Transforming data into decisions.*
