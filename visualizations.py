"""
Visualization Module
Creates various charts and plots for data analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class VisualizationManager:
    """Manages all visualizations for the dashboard"""
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_distribution(self, data, column_name, bins=20):
        """Plot distribution of a numeric column"""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(column_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {column_name}')
        plt.tight_layout()
        return fig
    
    def plot_categorical_counts(self, data, column_name, top_n=10):
        """Plot counts of categorical values"""
        counts = data.value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot.bar(ax=ax, color='orange', edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title(f'Top {top_n} {column_name}')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return None
        
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results):
        """Plot comparison of different models"""
        models = list(model_results.keys())
        test_r2 = [model_results[m]['test_r2'] for m in models]
        test_mae = [model_results[m]['test_mae'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # R² Score comparison
        ax1.bar(models, test_r2, color='green', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison - R² Score')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45)
        
        # MAE comparison
        ax2.bar(models, test_mae, color='red', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Model Comparison - MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_test, y_pred, model_name):
        """Plot predicted vs actual values"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name} - Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_dict, top_n=10):
        """Plot feature importance"""
        if not importance_dict:
            return None
        
        # Sort by importance and get top N
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(features)), importances, color='purple', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_test, y_pred, model_name):
        """Plot residuals (errors)"""
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name} - Residual Plot')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

