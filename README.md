# Sales Analytics Dashboard 🚀

An enterprise-grade, highly interactive **Sales Intelligence Platform** built with Streamlit. Designed with a luxury SaaS dark-mode aesthetic, this dashboard moves beyond static reporting to provide dynamic exploration, advanced analytics, anomaly detection, and predictive forecasting.

![Dashboard Preview](https://via.placeholder.com/1200x600.png?text=Executive+Command+Center)

---

## ✨ Features

- **Executive Command Center**: High-level KPIs, Health Scores, and Growth Scores evaluated against prior periods.
- **Interactive Business Explorer**: A Tableau/Power BI style module allowing dynamic dimension and metric selection to auto-generate the optimal visualizations (Sunbursts, Donuts, Grouped Bars).
- **Advanced Compare Mode**: Select two entities (e.g., North vs South) to generate side-by-side KPIs, performance deltas, and automated AI insights.
- **Drill-Down Breadcrumbs**: Click on charts to drill into the underlying data (e.g., clicking the North region filters the entire module down to North).
- **Business Scenario Simulator (What-If)**: Adjust business levers (e.g., Discount Rate, Demand Volume) using sliders to project instant revenue impacts via elasticity models.
- **Anomaly Detection**: Uses Isolation Forests to identify statistically significant outliers and anomalies in transactional data.
- **AI Executive Summary**: Generates a textual breakdown explaining the "Why" behind the data, identifying key drivers, best-selling products, and underperforming regions.
- **Premium SaaS Aesthetic**: Frosted glassmorphism, responsive hover micro-interactions, rich gradients, and an exclusive deep navy theme built entirely in CSS.

---

## 🛠️ Tech Stack

- **Frontend / Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Machine Learning / Stats**: [Scikit-Learn](https://scikit-learn.org/) (Isolation Forest), [SciPy](https://scipy.org/)
- **Styling**: Custom CSS (Inter Font, Flexbox, Glassmorphism)

---

## 💻 Installation Steps (Local Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sales-analytics-dashboard.git
   cd sales-analytics-dashboard
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

## ☁️ Deployment Instructions (Streamlit Community Cloud)

This project has been fully audited and prepared for zero-configuration deployment on **Streamlit Community Cloud**.

### Prerequisites
- A GitHub account.
- A Streamlit Community Cloud account (free).

### Steps to Deploy
1. **Push your code to GitHub**: Ensure all files (including `requirements.txt`, `Product-Sales-Region (1).xlsx`, `.streamlit/config.toml`, and `.py` files) are committed to your repository.
2. **Log into Streamlit Community Cloud**: Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. **Deploy a New App**:
   - Click **New app**.
   - Select your GitHub repository containing the dashboard.
   - Set the **Main file path** to `dashboard.py`.
4. **Deploy**: Click the **Deploy!** button. Streamlit will automatically read `requirements.txt`, install dependencies, and launch the dashboard.

### Cloud Deployment Notes
- **File Paths**: All data loading paths (e.g., loading `Product-Sales-Region (1).xlsx`) are configured securely using relative paths.
- **Theme Configuration**: The `.streamlit/config.toml` enforces the required dark mode parameters to ensure the premium CSS aesthetic renders flawlessly on the cloud.
- **Memory Optimization**: The `EXPECTED_MIN_ROWS` check ensures the data is loaded correctly, while `@st.cache_data` decorators prevent reloading data on every user interaction, ensuring snappy performance within Streamlit Cloud's memory limits.

---

## 📂 Project Structure

- `dashboard.py`: The main entry point and UI layout.
- `utils.py`: Centralized CSS styling engine, KPI logic, and text formatting.
- `explorer.py`: The Interactive Business Explorer logic (pivot tables, auto-charting, drill-down).
- `intelligence.py`: Advanced analytical models (AI Summaries, Top Movers, What-If Analysis, Anomalies).
- `data_loader.py` & `data_cleaning.py`: Ingestion and preprocessing pipelines.
- `feature_engineering.py`: Data enrichment.
- `requirements.txt`: Cloud dependencies.
- `.streamlit/config.toml`: Cloud configuration overrides.
