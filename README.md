<<<<<<< HEAD
# 🤖 Universal ML Dashboard

A comprehensive machine learning and data analysis tool that works with **any dataset** for regression tasks. Built with Python, Streamlit, and Scikit-learn.

## ✨ Features

### 📊 Data Overview
- Dataset statistics and information
- Column details and data types
- Statistical summaries (mean, median, std, etc.)
- Missing values analysis

### 📈 Exploratory Data Analysis
- **Correlation Matrix**: Visualize relationships between numeric features
- **Distribution Plots**: Analyze data distributions
- **Categorical Analysis**: Top N value counts for categorical columns
- **Missing Values Visualization**: Identify and analyze missing data

### 🤖 Machine Learning
- **Multiple Models**: Train and compare 4 different ML algorithms:
  - Random Forest Regressor
  - Linear Regression
  - Decision Tree Regressor
  - Gradient Boosting Regressor
- **Automatic Feature Selection**: Choose which columns to use as features
- **Flexible Target Selection**: Predict any numeric column
- **Real-time Predictions**: Make predictions with custom input values

### 🔍 Model Comparison
- **Performance Metrics**: R² Score, MAE, RMSE for all models
- **Visual Comparisons**: Side-by-side model performance charts
- **Prediction vs Actual Plots**: Visualize model accuracy
- **Residual Analysis**: Analyze prediction errors
- **Feature Importance**: Understand which features matter most

### 📤 Export Results
- Download predictions as CSV
- Export model metrics
- Save feature importance data

## 🚀 Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

2. **Upload your dataset:**
   - Click on the sidebar
   - Upload a CSV file
   - The app will automatically detect numeric and categorical columns

3. **Explore your data:**
   - Use the "Data Overview" tab to understand your dataset
   - Check "Exploratory Analysis" for visualizations

4. **Train ML models:**
   - Go to "Machine Learning" tab
   - Select your target column (what you want to predict)
   - Choose feature columns (predictors)
   - Click "Train All Models"

5. **Compare and analyze:**
   - View model comparisons in the "Model Comparison" tab
   - Check feature importance
   - Make predictions with custom inputs

6. **Export results:**
   - Download predictions, metrics, and feature importance from the "Export Results" tab

## 📋 Dataset Requirements

- **Format**: CSV file
- **Target Variable**: At least one numeric column (for regression)
- **Features**: Mix of numeric and categorical columns
- **No strict column names required**: Works with any dataset!

## 🏗️ Project Structure

```
.
├── main.py                 # Main Streamlit application
├── data_processor.py       # Data loading and preprocessing module
├── ml_models.py           # Machine learning models manager
├── visualizations.py      # Visualization functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Technologies

- **Python 3.8+**
- **Streamlit**: Interactive web interface
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization

## 📝 Code Architecture

The project follows **Object-Oriented Programming (OOP)** principles:

- **`DataProcessor`**: Handles data loading, cleaning, and preprocessing
- **`MLModelManager`**: Manages multiple ML models and their training
- **`VisualizationManager`**: Creates all charts and plots

## 🎯 Use Cases

This tool can be used for:
- **Price Prediction**: Predict prices (houses, cars, products, etc.)
- **Sales Forecasting**: Predict sales based on various factors
- **Performance Analysis**: Predict performance metrics
- **Any Regression Task**: Works with any numeric target variable

## 👥 Team

Developed by a team of 3 students as a semester project.

## 📄 License

This project is created for educational purposes.

---

**Made with ❤️ for Data Science**


=======
# Universal-Ml-Dashboard
This project focuses on creating a generic and reusable dashboard that works with any CSV dataset, allowing users to: - Explore and understand data - Perform exploratory data analysis (EDA) - Train multiple machine learning regression models - Compare model performance using standard evaluation metrics  Make predictions and export results
>>>>>>> 86659077f6629d098fb916ce851518386b3756e1
