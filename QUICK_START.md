# 🚀 Quick Start Guide

## Step 1: Install Dependencies

Open your terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

## Step 2: Run the Application

```bash
streamlit run main.py
```

The app will open in your browser automatically!

## Step 3: Upload Your Dataset

1. Click on the sidebar (left side)
2. Click "Browse files" under "Upload your CSV dataset"
3. Select any CSV file with numeric and categorical columns

## Step 4: Explore Your Data

- **Data Overview Tab**: See basic statistics and column information
- **Exploratory Analysis Tab**: View correlations and distributions

## Step 5: Train ML Models

1. Go to **"Machine Learning"** tab
2. Select your **target column** (what you want to predict - must be numeric)
3. Select **feature columns** (predictors - can be numeric or categorical)
4. Click **"🚀 Train All Models"**
5. Wait for training to complete (usually takes a few seconds)

## Step 6: Make Predictions

1. After training, scroll down to the prediction section
2. Enter values for your features
3. Click **"🔮 Predict"**
4. See your prediction!

## Step 7: Compare Models

1. Go to **"Model Comparison"** tab
2. View performance metrics for all models
3. See which model performs best
4. Check feature importance to understand what matters most

## Step 8: Export Results

1. Go to **"Export Results"** tab
2. Download predictions, metrics, or feature importance as CSV files

## 💡 Tips

- **Target Column**: Must be numeric (e.g., price, sales, temperature)
- **Feature Columns**: Can be numeric (e.g., size, age) or categorical (e.g., brand, category)
- **More Data = Better Models**: Try to have at least 50-100 rows for good results
- **Clean Data**: Remove obvious errors before uploading for best results

## 📊 Example Datasets You Can Use

- House prices (predict price from size, location, bedrooms, etc.)
- Car prices (predict price from brand, year, mileage, etc.)
- Sales data (predict sales from marketing spend, season, etc.)
- Student performance (predict score from study hours, attendance, etc.)
- Any dataset with numeric target and mixed features!

## ❓ Troubleshooting

**Problem**: "No numeric columns found"
- **Solution**: Make sure your CSV has at least one numeric column

**Problem**: "Error loading file"
- **Solution**: Check that your file is a valid CSV format

**Problem**: Models not training
- **Solution**: Make sure you selected both target and feature columns

**Problem**: Predictions seem wrong
- **Solution**: Check that your input values are within the range of your training data

---

Happy analyzing! 🎉

