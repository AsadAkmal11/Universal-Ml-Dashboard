"""
Machine Learning Models Module
Contains multiple ML models for comparison and prediction
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import warnings


class MLModelManager:
    """Manages multiple ML models for comparison"""
    
    def __init__(self, random_state=42):
        """
        Initialize ML Model Manager
        
        Args:
            random_state: Random seed for reproducibility (default: 42)
        """
        self.random_state = random_state
        # Initialize models with default hyperparameters
        # Hyperparameters will be updated via update_hyperparameters method
        self.default_hyperparams = {
            'Random Forest': {'n_estimators': 100},
            'Linear Regression': {},
            'Decision Tree': {'max_depth': None},
            'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1}
        }
        self.models = self._create_models(self.default_hyperparams)
        self.trained_models = {}
        self.model_scores = {}
        self.cv_scores = {}  # Store cross-validation results
        
    def _create_models(self, hyperparams):
        """Create model instances with specified hyperparameters"""
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=hyperparams['Random Forest'].get('n_estimators', 100),
                random_state=self.random_state,
                n_jobs=-1  # Use all CPU cores
            ),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=hyperparams['Decision Tree'].get('max_depth', None),
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=hyperparams['Gradient Boosting'].get('n_estimators', 100),
                learning_rate=hyperparams['Gradient Boosting'].get('learning_rate', 0.1),
                random_state=self.random_state
            )
        }
        return models
    
    def update_hyperparameters(self, hyperparams):
        """
        Update model hyperparameters
        
        Args:
            hyperparams: Dictionary with model names as keys and hyperparameter dicts as values
                Example: {'Decision Tree': {'max_depth': 10}, 'Random Forest': {'n_estimators': 200}}
        """
        for model_name, params in hyperparams.items():
            if model_name in self.default_hyperparams:
                self.default_hyperparams[model_name].update(params)
        # Recreate models with new hyperparameters
        self.models = self._create_models(self.default_hyperparams)
    
    def perform_cross_validation(self, X, y, k_folds=5):
        """
        Perform K-Fold Cross Validation on all models
        
        Args:
            X: Feature matrix
            y: Target vector
            k_folds: Number of folds (default: 5)
        
        Returns:
            Dictionary with CV results for each model
        """
        cv_results = {}
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            # Perform cross-validation for R², MAE, and RMSE
            cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            cv_mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1))
            
            cv_results[name] = {
                'mean_cv_r2': np.mean(cv_r2_scores),
                'std_cv_r2': np.std(cv_r2_scores),
                'mean_cv_mae': np.mean(cv_mae_scores),
                'std_cv_mae': np.std(cv_mae_scores),
                'mean_cv_rmse': np.mean(cv_rmse_scores),
                'std_cv_rmse': np.std(cv_rmse_scores),
                'cv_r2_scores': cv_r2_scores,
                'cv_mae_scores': cv_mae_scores,
                'cv_rmse_scores': cv_rmse_scores
            }
        
        self.cv_scores = cv_results
        return cv_results
    
    def detect_overfitting(self, train_r2, test_r2, threshold=0.1):
        """
        Detect potential overfitting
        
        Args:
            train_r2: Training R² score
            test_r2: Test R² score
            threshold: Minimum difference to flag as overfitting (default: 0.1)
        
        Returns:
            Boolean indicating if overfitting is detected
        """
        # Overfitting: high train R² but significantly lower test R²
        r2_diff = train_r2 - test_r2
        return r2_diff > threshold and train_r2 > 0.7
    
    def train_all_models(self, X_train, y_train, X_test, y_test, perform_cv=True, k_folds=5):
        """
        Train all models and evaluate them
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            perform_cv: Whether to perform cross-validation (default: True)
            k_folds: Number of folds for CV (default: 5)
        
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        # Perform cross-validation on training data if requested
        if perform_cv:
            self.perform_cross_validation(X_train, y_train, k_folds=k_folds)
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Detect overfitting
            is_overfitting = self.detect_overfitting(train_r2, test_r2)
            
            # Get CV results if available
            cv_data = self.cv_scores.get(name, {})
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'y_pred_test': y_pred_test,
                'y_test': y_test,
                'is_overfitting': is_overfitting,
                'r2_diff': train_r2 - test_r2,
                # Cross-validation metrics
                'mean_cv_r2': cv_data.get('mean_cv_r2', None),
                'std_cv_r2': cv_data.get('std_cv_r2', None),
                'mean_cv_mae': cv_data.get('mean_cv_mae', None),
                'mean_cv_rmse': cv_data.get('mean_cv_rmse', None)
            }
            
            self.model_scores[name] = results[name]
        
        return results
    
    def get_best_model(self):
        """
        Get the best model based on improved selection logic:
        1. Primary criterion: Highest Test R² score
        2. Tie-breaker: Lowest Test RMSE (if R² scores are very close)
        
        Returns:
            Tuple of (model_name, model_instance) or (None, None) if no models trained
        """
        if not self.model_scores:
            return None, None
        
        # Find maximum test R²
        max_test_r2 = max(self.model_scores[m]['test_r2'] for m in self.model_scores.keys())
        
        # Find all models with R² within 0.01 of the maximum (for tie-breaking)
        candidates = {
            name: scores for name, scores in self.model_scores.items()
            if abs(scores['test_r2'] - max_test_r2) < 0.01
        }
        
        if len(candidates) == 1:
            # Only one candidate, return it
            best_name = list(candidates.keys())[0]
        else:
            # Multiple candidates with similar R², use RMSE as tie-breaker
            # Lower RMSE is better
            best_name = min(candidates.keys(), 
                          key=lambda x: candidates[x]['test_rmse'])
        
        return best_name, self.trained_models[best_name]
    
    def predict(self, model_name, X):
        """Make prediction using specified model"""
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            # For linear regression - use absolute coefficients as importance
            coef = model.coef_
            return dict(zip(feature_names, np.abs(coef)))
        else:
            return None
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk using joblib
        
        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model
        """
        import joblib
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filepath)
        else:
            raise ValueError(f"Model {model_name} not found or not trained")
    
    def load_model(self, model_name, filepath):
        """
        Load a model from disk using joblib
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model file
        """
        import joblib
        self.trained_models[model_name] = joblib.load(filepath)

