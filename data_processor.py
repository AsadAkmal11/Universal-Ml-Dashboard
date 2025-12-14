"""
Data Processing Module
Handles data loading, cleaning, and preprocessing for any dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import warnings


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_columns = []
        self.categorical_columns = []
        self.target_column = None
        self.scaler_fitted = False
        self.numeric_feature_indices = None  # Track which columns are numeric for scaling
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def auto_detect_columns(self, df):
        """Automatically detect numeric and categorical columns"""
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from feature lists if it exists
        if self.target_column and self.target_column in self.numeric_columns:
            self.numeric_columns.remove(self.target_column)
        if self.target_column and self.target_column in self.categorical_columns:
            self.categorical_columns.remove(self.target_column)
            
        return self.numeric_columns, self.categorical_columns
    
    def clean_data(self, df):
        """Clean and preprocess data"""
        df = df.copy()
        
        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert numeric columns - extract numbers from strings if needed
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to extract numeric values from strings
                numeric_values = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
                if numeric_values.notna().sum() > len(df) * 0.5:  # If >50% are numeric
                    df[col] = pd.to_numeric(numeric_values, errors='coerce')
        
        return df
    
    def prepare_ml_data(self, df, target_column, feature_columns=None, fit_scaler=False, apply_scaling=False):
        """
        Prepare data for machine learning
        
        Args:
            df: DataFrame with data
            target_column: Name of target column
            feature_columns: List of feature column names
            fit_scaler: If True, fit the scaler on this data (should be training data)
            apply_scaling: If True, apply scaling to numeric features
        
        Returns:
            X_encoded: Preprocessed feature matrix
            y: Target vector
            feature_columns: List of feature column names
        """
        self.target_column = target_column
        
        # Auto-detect columns if not specified
        if feature_columns is None:
            self.auto_detect_columns(df)
            feature_columns = self.numeric_columns + self.categorical_columns
        
        # Remove target from features
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_column])
        
        # Prepare features
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Identify which columns are numeric (for scaling)
        numeric_feature_cols = [col for col in feature_columns if col in self.numeric_columns]
        categorical_feature_cols = [col for col in feature_columns if col in self.categorical_columns]
        
        # Encode categorical variables with robust handling
        X_encoded = X.copy()
        for col in categorical_feature_cols:
            if col in X_encoded.columns:
                if col not in self.label_encoders:
                    # Fit encoder on training data
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str).fillna('Unknown'))
                else:
                    # Transform with handling of unseen categories
                    known_classes = set(self.label_encoders[col].classes_)
                    X_encoded[col] = X[col].astype(str).fillna('Unknown').apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in known_classes else -1  # Use -1 for unknown categories
                    )
                    # Warn if unknown categories found
                    unknown_count = (X_encoded[col] == -1).sum()
                    if unknown_count > 0 and fit_scaler:
                        warnings.warn(f"Found {unknown_count} unseen categories in '{col}' during training. "
                                    f"They will be encoded as -1.")
        
        # Store numeric feature indices for scaling
        if numeric_feature_cols:
            # Get column indices after encoding (maintain order)
            self.numeric_feature_indices = [feature_columns.index(col) for col in numeric_feature_cols]
        else:
            self.numeric_feature_indices = []
        
        # Apply feature scaling to numeric features (if requested and numeric features exist)
        if apply_scaling and self.numeric_feature_indices and len(numeric_feature_cols) > 0:
            if fit_scaler:
                # Fit scaler on training data only
                X_numeric = X_encoded.iloc[:, self.numeric_feature_indices].values
                self.scaler.fit(X_numeric)
                self.scaler_fitted = True
                # Transform training data
                X_scaled = self.scaler.transform(X_numeric)
                X_encoded.iloc[:, self.numeric_feature_indices] = X_scaled
            elif self.scaler_fitted:
                # Transform test/prediction data using fitted scaler
                X_numeric = X_encoded.iloc[:, self.numeric_feature_indices].values
                X_scaled = self.scaler.transform(X_numeric)
                X_encoded.iloc[:, self.numeric_feature_indices] = X_scaled
            else:
                warnings.warn("Scaler not fitted. Cannot apply scaling. Use fit_scaler=True on training data first.")
        
        # Drop rows with missing values in features
        mask = ~(X_encoded.isna().any(axis=1))
        X_encoded = X_encoded[mask]
        y = y[mask]
        
        return X_encoded, y, feature_columns
    
    def encode_categorical_input(self, input_dict, feature_columns, apply_scaling=False):
        """
        Encode categorical inputs for prediction with robust handling
        
        Args:
            input_dict: Dictionary of feature values
            feature_columns: List of feature column names in correct order
            apply_scaling: If True, apply scaling to numeric features
        
        Returns:
            encoded: Dictionary of encoded values
        """
        encoded = {}
        for col in feature_columns:
            if col in self.categorical_columns and col in self.label_encoders:
                value = str(input_dict.get(col, 'Unknown'))
                if value in self.label_encoders[col].classes_:
                    encoded[col] = self.label_encoders[col].transform([value])[0]
                else:
                    encoded[col] = -1  # Unknown category - handled gracefully
                    warnings.warn(f"Unknown category '{value}' in column '{col}'. Using encoding -1.")
            else:
                encoded[col] = input_dict.get(col, 0)
        
        # Apply scaling to numeric features if needed
        if apply_scaling and self.scaler_fitted and self.numeric_feature_indices:
            # Create array in correct feature order
            feature_array = np.array([[encoded[col] for col in feature_columns]])
            # Scale only numeric features
            feature_array_scaled = feature_array.copy()
            numeric_values = feature_array[:, self.numeric_feature_indices]
            scaled_numeric = self.scaler.transform(numeric_values)
            feature_array_scaled[:, self.numeric_feature_indices] = scaled_numeric
            # Update encoded dict with scaled values
            for idx, col in enumerate(feature_columns):
                if col in [feature_columns[i] for i in self.numeric_feature_indices]:
                    encoded[col] = feature_array_scaled[0, idx]
        
        return encoded
    
    def get_data_summary(self, df):
        """Get statistical summary of the dataset"""
        summary = {
            'shape': df.shape,
            'numeric_summary': df.describe() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None,
            'categorical_summary': df.describe(include=['object']) if len(df.select_dtypes(include=['object']).columns) > 0 else None,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        return summary

