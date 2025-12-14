"""
Universal ML Dashboard - A comprehensive machine learning and data analysis tool
Works with any dataset for regression tasks
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import io

# Import our custom modules
from data_processor import DataProcessor
from ml_models import MLModelManager
from visualizations import VisualizationManager

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Universal ML Dashboard", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'ml_manager' not in st.session_state:
    st.session_state.ml_manager = MLModelManager(random_state=42)
if 'viz_manager' not in st.session_state:
    st.session_state.viz_manager = VisualizationManager()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Title and Header
st.title("🤖 Universal ML Dashboard")
st.markdown("**A Comprehensive Machine Learning & Data Analysis Tool**")
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset", 
    type=["csv"],
    help="Upload any CSV file with numeric and categorical columns"
)

if uploaded_file:
    try:
        # Load and clean data
        df = st.session_state.data_processor.load_data(uploaded_file)
        df = st.session_state.data_processor.clean_data(df)
        st.session_state.df = df
        st.session_state.data_processor.auto_detect_columns(df)
        
        st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display column info in sidebar
        st.sidebar.subheader("📊 Dataset Info")
        st.sidebar.write(f"**Numeric columns:** {len(st.session_state.data_processor.numeric_columns)}")
        st.sidebar.write(f"**Categorical columns:** {len(st.session_state.data_processor.categorical_columns)}")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.stop()
else:
    st.info("👈 Please upload a CSV file from the sidebar to get started!")
    st.stop()

df = st.session_state.df

# ==================== MAIN CONTENT ====================

# Tab navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview", 
    "📈 Exploratory Analysis", 
    "🤖 Machine Learning", 
    "🔍 Model Comparison",
    "📤 Export Results",
    "ℹ️ About"
])

# ==================== TAB 1: DATA OVERVIEW ====================
with tab1:
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("📋 Data Preview")
    num_rows = st.slider("Number of rows to display", 5, 100, 20)
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    st.subheader("📝 Column Information")
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    if len(st.session_state.data_processor.categorical_columns) > 0:
        st.subheader("📋 Categorical Summary")
        cat_summary = df[st.session_state.data_processor.categorical_columns].describe()
        st.dataframe(cat_summary, use_container_width=True)

# ==================== TAB 2: EXPLORATORY ANALYSIS ====================
with tab2:
    st.header("📈 Exploratory Data Analysis")
    
    # Correlation Matrix
    if len(st.session_state.data_processor.numeric_columns) >= 2:
        st.subheader("🔗 Correlation Matrix")
        corr_fig = st.session_state.viz_manager.plot_correlation_matrix(df)
        if corr_fig:
            st.pyplot(corr_fig)
            plt.close()
    
    # Distribution plots
    st.subheader("📊 Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.data_processor.numeric_columns:
            selected_num_col = st.selectbox(
                "Select numeric column for distribution",
                st.session_state.data_processor.numeric_columns
            )
            bins = st.slider("Number of bins", 10, 50, 20)
            dist_fig = st.session_state.viz_manager.plot_distribution(
                df[selected_num_col].dropna(), selected_num_col, bins
            )
            st.pyplot(dist_fig)
            plt.close()
    
    with col2:
        if st.session_state.data_processor.categorical_columns:
            selected_cat_col = st.selectbox(
                "Select categorical column",
                st.session_state.data_processor.categorical_columns
            )
            top_n = st.slider("Top N values", 5, 20, 10)
            cat_fig = st.session_state.viz_manager.plot_categorical_counts(
                df[selected_cat_col], selected_cat_col, top_n
            )
            st.pyplot(cat_fig)
            plt.close()
    
    # Missing values visualization
    st.subheader("🔍 Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Percentage': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found!")

# ==================== TAB 3: MACHINE LEARNING ====================
with tab3:
    st.header("🤖 Machine Learning Model Training")
    
    # Target and feature selection
    st.subheader("🎯 Configure Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target column selection
        if st.session_state.data_processor.numeric_columns:
            target_column = st.selectbox(
                "Select Target Column (what to predict)",
                st.session_state.data_processor.numeric_columns,
                help="Choose the numeric column you want to predict"
            )
        else:
            st.error("No numeric columns found for target variable!")
            st.stop()
    
    with col2:
        # Feature selection
        all_features = st.session_state.data_processor.numeric_columns + \
                      st.session_state.data_processor.categorical_columns
        if target_column in all_features:
            all_features.remove(target_column)
        
        feature_columns = st.multiselect(
            "Select Feature Columns (predictors)",
            all_features,
            default=all_features[:min(5, len(all_features))],
            help="Select columns to use as features for prediction"
        )
    
    if not feature_columns:
        st.warning("⚠️ Please select at least one feature column!")
        st.stop()
    
    # Train/Test split ratio
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    # Feature scaling option
    st.subheader("⚙️ Advanced Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_scaling = st.checkbox(
            "Apply Feature Scaling",
            value=True,
            help="StandardScaler for numeric features. Tree-based models (Decision Tree, Random Forest, Gradient Boosting) don't require scaling, but it helps Linear Regression."
        )
    
    with col2:
        perform_cv = st.checkbox(
            "Perform Cross-Validation",
            value=True,
            help="K-Fold Cross Validation (k=5) for more robust model evaluation"
        )
    
    with col3:
        k_folds = st.number_input(
            "CV Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
    
    # Hyperparameter tuning section
    st.subheader("🎛️ Hyperparameter Tuning")
    st.caption("Adjust model hyperparameters (optional - defaults are usually good)")
    
    hyperparams = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Decision Tree**")
        dt_max_depth = st.slider(
            "Max Depth",
            min_value=1,
            max_value=30,
            value=10,
            key="dt_depth",
            help="Maximum depth of the tree (None = unlimited, but 10 is a good default)"
        )
        hyperparams['Decision Tree'] = {'max_depth': dt_max_depth if dt_max_depth < 30 else None}
    
    with col2:
        st.markdown("**Random Forest**")
        rf_n_estimators = st.slider(
            "N Estimators",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="rf_estimators",
            help="Number of trees in the forest"
        )
        hyperparams['Random Forest'] = {'n_estimators': rf_n_estimators}
    
    with col3:
        st.markdown("**Gradient Boosting**")
        gb_n_estimators = st.slider(
            "N Estimators",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="gb_estimators",
            help="Number of boosting stages"
        )
        gb_learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            key="gb_lr",
            help="Learning rate shrinks the contribution of each tree"
        )
        hyperparams['Gradient Boosting'] = {
            'n_estimators': gb_n_estimators,
            'learning_rate': gb_learning_rate
        }
    
    # Update hyperparameters
    st.session_state.ml_manager.update_hyperparameters(hyperparams)
    
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take a moment."):
            try:
                # Prepare data with encoding but WITHOUT scaling first
                # This ensures categorical encoding is done on full dataset
                X_full, y_full, _ = st.session_state.data_processor.prepare_ml_data(
                    df, target_column, feature_columns, fit_scaler=False, apply_scaling=False
                )
                
                if len(X_full) == 0:
                    st.error("No valid data after preprocessing. Please check your data.")
                    st.stop()
                
                # Split data with fixed random_state for reproducibility
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=test_size, random_state=42
                )
                
                # Apply feature scaling if enabled
                # CRITICAL ML BEST PRACTICE: Fit scaler ONLY on training data, then transform both train and test
                if use_scaling and st.session_state.data_processor.numeric_feature_indices:
                    # Get numeric column indices
                    numeric_indices = st.session_state.data_processor.numeric_feature_indices
                    
                    # Extract numeric features from training data
                    X_train_numeric = X_train.iloc[:, numeric_indices].values
                    
                    # Fit scaler on training data only (CRITICAL: never fit on test data)
                    st.session_state.data_processor.scaler.fit(X_train_numeric)
                    st.session_state.data_processor.scaler_fitted = True
                    
                    # Transform training data
                    X_train_scaled = st.session_state.data_processor.scaler.transform(X_train_numeric)
                    X_train.iloc[:, numeric_indices] = X_train_scaled
                    
                    # Transform test data using the fitted scaler (CRITICAL: don't fit on test data)
                    X_test_numeric = X_test.iloc[:, numeric_indices].values
                    X_test_scaled = st.session_state.data_processor.scaler.transform(X_test_numeric)
                    X_test.iloc[:, numeric_indices] = X_test_scaled
                
                # Note: For tree-based models (Decision Tree, Random Forest, Gradient Boosting),
                # scaling is not necessary as they are scale-invariant. However, applying scaling
                # doesn't hurt and maintains consistency. Linear Regression benefits significantly from scaling.
                
                X_train_processed = X_train
                X_test_processed = X_test
                y_train_processed = y_train
                y_test_processed = y_test
                
                # Train models
                results = st.session_state.ml_manager.train_all_models(
                    X_train_processed, y_train_processed, 
                    X_test_processed, y_test_processed,
                    perform_cv=perform_cv,
                    k_folds=k_folds
                )
                
                st.session_state.trained = True
                st.session_state.X_test = X_test_processed
                st.session_state.y_test = y_test_processed
                st.session_state.feature_columns = feature_columns
                st.session_state.target_column = target_column
                st.session_state.use_scaling = use_scaling  # Store scaling preference for predictions
                
                st.success("✅ All models trained successfully!")
                
                # Display results with overfitting warnings
                st.subheader("📊 Model Performance Summary")
                
                # Build results dataframe
                results_data = {
                    'Model': list(results.keys()),
                    'Train R²': [results[m]['train_r2'] for m in results.keys()],
                    'Test R²': [results[m]['test_r2'] for m in results.keys()],
                    'Train MAE': [results[m]['train_mae'] for m in results.keys()],
                    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
                    'Train RMSE': [results[m]['train_rmse'] for m in results.keys()],
                    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()]
                }
                
                # Add cross-validation metrics if available
                if perform_cv:
                    results_data['Mean CV R²'] = [
                        results[m].get('mean_cv_r2', None) for m in results.keys()
                    ]
                    results_data['Mean CV MAE'] = [
                        results[m].get('mean_cv_mae', None) for m in results.keys()
                    ]
                    results_data['Mean CV RMSE'] = [
                        results[m].get('mean_cv_rmse', None) for m in results.keys()
                    ]
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df.round(4), use_container_width=True)
                
                # Display overfitting warnings
                overfitting_models = [name for name, res in results.items() if res.get('is_overfitting', False)]
                if overfitting_models:
                    st.warning("⚠️ **Overfitting Detected**: The following models show signs of overfitting "
                             f"(high Train R² but significantly lower Test R²): {', '.join(overfitting_models)}")
                    for model_name in overfitting_models:
                        r2_diff = results[model_name]['r2_diff']
                        st.caption(f"  • {model_name}: Train-Test R² difference = {r2_diff:.4f}")
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
    
    # Prediction section
    if st.session_state.trained:
        st.subheader("🔮 Make Predictions")
        
        # Get best model
        best_model_name, _ = st.session_state.ml_manager.get_best_model()
        selected_model = st.selectbox(
            "Select Model for Prediction",
            list(st.session_state.ml_manager.models.keys()),
            index=list(st.session_state.ml_manager.models.keys()).index(best_model_name) if best_model_name else 0
        )
        
        st.info(f"💡 Best model: **{best_model_name}** (highest R² score)")
        
        # Create input form
        st.write("Enter feature values:")
        input_dict = {}
        
        cols = st.columns(min(3, len(st.session_state.feature_columns)))
        for idx, col_name in enumerate(st.session_state.feature_columns):
            with cols[idx % len(cols)]:
                if col_name in st.session_state.data_processor.categorical_columns:
                    unique_vals = df[col_name].dropna().unique()
                    input_dict[col_name] = st.selectbox(f"{col_name}", unique_vals)
                else:
                    min_val = float(df[col_name].min())
                    max_val = float(df[col_name].max())
                    median_val = float(df[col_name].median())
                    input_dict[col_name] = st.number_input(
                        f"{col_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val
                    )
        
        if st.button("🔮 Predict", type="primary"):
            try:
                # Encode input with scaling if it was used during training
                # Check if scaling was used (we'll track this in session state)
                use_scaling_for_pred = st.session_state.get('use_scaling', True)
                encoded_input = st.session_state.data_processor.encode_categorical_input(
                    input_dict, st.session_state.feature_columns, apply_scaling=use_scaling_for_pred
                )
                
                # Create input array in correct order
                input_array = np.array([[encoded_input[col] for col in st.session_state.feature_columns]])
                
                # Predict
                prediction = st.session_state.ml_manager.predict(selected_model, input_array)[0]
                
                st.success(f"🎯 **Predicted {st.session_state.target_column}: {prediction:,.2f}**")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# ==================== TAB 4: MODEL COMPARISON ====================
with tab4:
    st.header("📊 Model Comparison & Analysis")
    
    if not st.session_state.trained:
        st.info("👈 Please train models first in the 'Machine Learning' tab!")
    else:
        # Model comparison chart
        results = st.session_state.ml_manager.model_scores
        if results:
            st.subheader("📈 Model Performance Comparison")
            comp_fig = st.session_state.viz_manager.plot_model_comparison(results)
            st.pyplot(comp_fig)
            plt.close()
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics")
            metrics_cols = ['train_r2', 'test_r2', 'train_mae', 'test_mae', 'train_rmse', 'test_rmse']
            
            # Add CV metrics if available
            if any(results[m].get('mean_cv_r2') is not None for m in results.keys()):
                metrics_cols.extend(['mean_cv_r2', 'mean_cv_mae', 'mean_cv_rmse'])
            
            metrics_df = pd.DataFrame(results).T
            available_cols = [col for col in metrics_cols if col in metrics_df.columns]
            st.dataframe(metrics_df[available_cols].round(4), use_container_width=True)
            
            # Display cross-validation results if available
            if any(results[m].get('mean_cv_r2') is not None for m in results.keys()):
                st.subheader("📊 Cross-Validation Results (K-Fold)")
                cv_data = {
                    'Model': list(results.keys()),
                    'Mean CV R²': [results[m].get('mean_cv_r2', None) for m in results.keys()],
                    'Std CV R²': [results[m].get('std_cv_r2', None) for m in results.keys()],
                    'Mean CV MAE': [results[m].get('mean_cv_mae', None) for m in results.keys()],
                    'Mean CV RMSE': [results[m].get('mean_cv_rmse', None) for m in results.keys()]
                }
                cv_df = pd.DataFrame(cv_data)
                st.dataframe(cv_df.round(4), use_container_width=True)
                st.caption("Cross-validation provides a more robust estimate of model performance by testing on multiple train/test splits.")
            
            # Best model info with improved selection logic
            best_name, _ = st.session_state.ml_manager.get_best_model()
            best_r2 = results[best_name]['test_r2']
            best_rmse = results[best_name]['test_rmse']
            st.success(f"🏆 **Best Model: {best_name}** | Test R² = {best_r2:.4f} | Test RMSE = {best_rmse:.4f}")
            st.caption("Selection: Highest Test R², with Test RMSE as tie-breaker for similar R² scores.")
            
            # Overfitting warnings
            overfitting_models = [name for name, res in results.items() if res.get('is_overfitting', False)]
            if overfitting_models:
                st.warning("⚠️ **Overfitting Alert**: The following models may be overfitting: "
                         f"{', '.join(overfitting_models)}. Consider regularization or simpler models.")
            
            # Prediction vs Actual plots
            st.subheader("📉 Prediction vs Actual Values")
            selected_model_viz = st.selectbox(
                "Select model to visualize",
                list(results.keys())
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                pred_actual_fig = st.session_state.viz_manager.plot_prediction_vs_actual(
                    results[selected_model_viz]['y_test'],
                    results[selected_model_viz]['y_pred_test'],
                    selected_model_viz
                )
                st.pyplot(pred_actual_fig)
                plt.close()
            
            with col2:
                residual_fig = st.session_state.viz_manager.plot_residuals(
                    results[selected_model_viz]['y_test'],
                    results[selected_model_viz]['y_pred_test'],
                    selected_model_viz
                )
                st.pyplot(residual_fig)
                plt.close()
            
            # Feature importance
            st.subheader("🎯 Feature Importance")
            importance = st.session_state.ml_manager.get_feature_importance(
                selected_model_viz,
                st.session_state.feature_columns
            )
            
            if importance:
                top_n_features = st.slider("Top N features", 5, len(st.session_state.feature_columns), min(10, len(st.session_state.feature_columns)))
                importance_fig = st.session_state.viz_manager.plot_feature_importance(importance, top_n_features)
                if importance_fig:
                    st.pyplot(importance_fig)
                    plt.close()
                
                # Feature importance table
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

# ==================== TAB 5: EXPORT RESULTS ====================
with tab5:
    st.header("📤 Export Results")
    
    if not st.session_state.trained:
        st.info("👈 Please train models first to export results!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Predictions")
            if st.button("Download Predictions CSV"):
                results = st.session_state.ml_manager.model_scores
                best_name, _ = st.session_state.ml_manager.get_best_model()
                
                predictions_df = pd.DataFrame({
                    'Actual': results[best_name]['y_test'],
                    'Predicted': results[best_name]['y_pred_test'],
                    'Error': results[best_name]['y_test'] - results[best_name]['y_pred_test']
                })
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("📈 Export Model Metrics")
            if st.button("Download Metrics CSV"):
                results = st.session_state.ml_manager.model_scores
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Train_R2': [results[m]['train_r2'] for m in results.keys()],
                    'Test_R2': [results[m]['test_r2'] for m in results.keys()],
                    'Train_MAE': [results[m]['train_mae'] for m in results.keys()],
                    'Test_MAE': [results[m]['test_mae'] for m in results.keys()],
                    'Train_RMSE': [results[m]['train_rmse'] for m in results.keys()],
                    'Test_RMSE': [results[m]['test_rmse'] for m in results.keys()]
                })
                
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="model_metrics.csv",
                    mime="text/csv"
                )
        
        st.subheader("📋 Export Feature Importance")
        if st.button("Download Feature Importance"):
            best_name, _ = st.session_state.ml_manager.get_best_model()
            importance = st.session_state.ml_manager.get_feature_importance(
                best_name,
                st.session_state.feature_columns
            )
            
            if importance:
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)
                
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )
        
        # Model persistence section
        st.subheader("💾 Model Persistence")
        st.caption("Save and load trained models (optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Save Model**")
            if st.session_state.trained:
                model_to_save = st.selectbox(
                    "Select model to save",
                    list(st.session_state.ml_manager.trained_models.keys()),
                    key="save_model_select"
                )
                if st.button("💾 Save Model", key="save_btn"):
                    try:
                        import joblib
                        import os
                        os.makedirs("saved_models", exist_ok=True)
                        filepath = f"saved_models/{model_to_save.replace(' ', '_')}.joblib"
                        st.session_state.ml_manager.save_model(model_to_save, filepath)
                        st.success(f"✅ Model saved to {filepath}")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
            else:
                st.info("Train models first to save them")
        
        with col2:
            st.markdown("**Load Model**")
            import os
            if os.path.exists("saved_models"):
                saved_models = [f for f in os.listdir("saved_models") if f.endswith(".joblib")]
                if saved_models:
                    model_to_load = st.selectbox(
                        "Select model to load",
                        saved_models,
                        key="load_model_select"
                    )
                    if st.button("📂 Load Model", key="load_btn"):
                        try:
                            filepath = os.path.join("saved_models", model_to_load)
                            model_name = model_to_load.replace(".joblib", "").replace("_", " ")
                            st.session_state.ml_manager.load_model(model_name, filepath)
                            st.success(f"✅ Model loaded: {model_name}")
                            st.info("Note: Loaded models can be used for predictions, but won't appear in comparison tables until you retrain.")
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
                else:
                    st.info("No saved models found")
            else:
                st.info("No saved models directory found")

# ==================== TAB 6: ABOUT ====================
with tab6:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🤖 Universal ML Dashboard
    
    A comprehensive machine learning and data analysis tool that works with **any dataset**!
    
    #### ✨ Features:
    - **📊 Data Overview**: Explore your dataset with statistical summaries
    - **📈 Exploratory Analysis**: Visualize correlations, distributions, and patterns
    - **🤖 Machine Learning**: Train multiple ML models (Random Forest, Linear Regression, Decision Tree, Gradient Boosting)
    - **🔍 Model Comparison**: Compare model performance with detailed metrics
    - **📤 Export Results**: Download predictions, metrics, and feature importance
    
    #### 🛠️ Technologies Used:
    - **Python** with Object-Oriented Programming
    - **Streamlit** for interactive web interface
    - **Scikit-learn** for machine learning
    - **Pandas & NumPy** for data processing
    - **Matplotlib & Seaborn** for visualizations
    
    #### 📝 How to Use:
    1. Upload your CSV dataset
    2. Explore the data in the "Data Overview" tab
    3. Analyze patterns in "Exploratory Analysis"
    4. Train ML models in the "Machine Learning" tab
    5. Compare models and view feature importance
    6. Export your results!
    
    #### 👥 Project Team:
    - Developed by a team of 3 students
    - Semester Project
    
    ---
    **Made with ❤️ for Data Science**
    """)
    
    st.markdown("---")