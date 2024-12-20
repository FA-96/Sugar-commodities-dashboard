# Sugar Commodities Dashboard

## **Overview**
This repository contains the code for a proof-of-concept dashboard that forecasts sugar commodity prices using machine learning. The dashboard integrates historical weather data, sugar production data, and global sugar price data to predict future prices and display key trends interactively.

---

## **Features**
1. **Historical and Forecasted Prices**:
   - Predicts daily sugar prices based on historical data using a Linear Regression model.
   - Includes adjusted prices (forecasted prices plus client-defined premiums and costs).

2. **Interactive Visualizations**:
   - Line chart comparing forecasted and adjusted prices.
   - Date slider to navigate through historical and forecasted prices.

3. **Future Price Predictions**:
   - Forecasts sugar prices for up to 30 days beyond the available data.

4. **User-Friendly Interface**:
   - Built with Streamlit for an intuitive and interactive user experience.

---

## **Data Sources**
- **Weather Data**:
  - Daily average temperatures and precipitation for Delhi, Lucknow, and Mumbai (1990–2022).
- **Production Data**:
  - Annual cane sugar production data for Brazil and India.
- **Sugar Prices**:
  - Monthly historical global sugar prices (No.11) from 1990 onwards.

---

## **Pipeline and Modeling**
1. **Data Preprocessing**:
   - Resampling and interpolating data to align daily weather, production, and price data.
   - Handling missing values with mean imputation.

2. **Feature Engineering**:
   - Combining weather and production data into a unified dataset.
   - Generating extrapolated features for future forecasting.

3. **Model**:
   - Linear Regression was chosen for simplicity and computational efficiency.

4. **Evaluation**:
   - The model is evaluated using Mean Absolute Error (MAE), demonstrating reasonable accuracy given the dataset constraints.

---

## **How to Run the Dashboard**

### **Local Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/FA-96/Sugar-commodities-dashboard.git
   cd Sugar-commodities-dashboard
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run Dashboardapp.py
   ```
4. Open the provided local URL in your browser to access the dashboard.

### **Streamlit Cloud**
The app is deployed on **Streamlit Cloud**. Access the live dashboard here:
[Live Dashboard Link](https://share.streamlit.io/FA-96/Sugar-commodities-dashboard)

---

## **File Structure**
```
Sugar-commodities-dashboard/
├── Dashboardapp.py               # Main Streamlit app code
├── psd_sugar.csv                 # Sugar production data
├── Delhi_NCR_1990_2022_Safdarjung.csv # Weather data for Delhi
├── Lucknow_1990_2022.csv         # Weather data for Lucknow
├── Mumbai_1990_2022_Santacruz.csv # Weather data for Mumbai
├── PSUGAISAUSDM.csv              # Global sugar price data
├── requirements.txt              # Python dependencies
└── README.md                     # Repository documentation
```

---

## **Limitations**
1. **Granularity Mismatch**:
   - Weather data is daily, production data is annual, and sugar price data is monthly. Interpolation was used to align the data.
2. **Simplistic Model**:
   - A Linear Regression model was used. Advanced models like Random Forest or Neural Networks could improve accuracy.
3. **Future Extrapolation Assumptions**:
   - Future weather and production data are extrapolated based on the latest available data, which assumes stability or seasonal patterns.

---

## **Future Improvements**
1. **Enhanced Models**:
   - Incorporate more advanced machine learning models for better accuracy.
2. **Real-Time Data**:
   - Integrate real-time weather and production data for dynamic forecasting.
3. **Expanded Features**:
   - Include macroeconomic factors like currency exchange rates, trade policies, and geopolitical events.

---

## **Contributors**
- **[Your Name](https://github.com/FA-96)**: Project Lead & Developer

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
