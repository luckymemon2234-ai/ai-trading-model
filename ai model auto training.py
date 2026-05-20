"""
XAU/USD LSTM Price Prediction Model - Production Version
========================================================
Advanced machine learning system for gold price forecasting with:
- Comprehensive data validation and preprocessing
- Advanced feature engineering
- Model versioning and checkpointing
- Performance monitoring and logging
- Hyperparameter optimization ready
- Production deployment features
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration management"""
    
    # Data Configuration
    DATA_FILE = "XAU_1Month_data.csv"
    DATE_COLUMN = "Date"
    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    TARGET_COLUMN = 'Close'
    
    # Model Parameters
    WINDOW_SIZE = 100
    PREDICTION_HORIZON = 10
    TRAIN_TEST_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    
    # Architecture
    LSTM_UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.3
    RECURRENT_DROPOUT = 0.2
    L1_REG = 1e-5
    L2_REG = 1e-4
    USE_BIDIRECTIONAL = True
    USE_BATCH_NORM = True
    
    # Training Configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7
    
    # Scaling Method
    SCALER_TYPE = 'minmax'  # 'minmax' or 'robust'
    
    # Directories
    MODEL_DIR = "models"
    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # File Names
    MODEL_NAME = "xauusd_lstm_production.keras"
    SCALER_NAME = "scaler.pkl"
    CONFIG_NAME = "model_config.json"
    METRICS_NAME = "training_metrics.csv"
    
    # Feature Engineering
    ADD_TECHNICAL_INDICATORS = True
    INDICATORS = {
        'SMA': [5, 10, 20],
        'EMA': [5, 10, 20],
        'RSI': [14],
        'MACD': True,
        'BOLLINGER': [20, 2],
        'ATR': [14],
        'VOLATILITY': [10, 20]
    }
    
    # Logging
    LOG_LEVEL = logging.INFO
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.MODEL_DIR, cls.LOG_DIR, cls.CHECKPOINT_DIR, cls.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath: str):
        """Save configuration to JSON"""
        config_dict = {k: v for k, v in cls.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with file and console handlers"""
    Config.create_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("XAU/USD LSTM Production Training Pipeline Started")
    logger.info("="*80)
    
    return logger


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

class DataLoader:
    """Handle data loading and validation"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        self.logger.info(f"Loading data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Try different separators
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(filepath, sep=sep)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        self.logger.info(f"Columns: {df.columns.tolist()}")
        
        # Validate required columns
        self._validate_columns(df)
        
        # Parse dates
        df[Config.DATE_COLUMN] = pd.to_datetime(df[Config.DATE_COLUMN])
        df = df.sort_values(Config.DATE_COLUMN).reset_index(drop=True)
        
        # Convert features to float
        for col in Config.FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Data quality checks
        self._quality_checks(df)
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame):
        """Validate presence of required columns"""
        required = [Config.DATE_COLUMN] + Config.FEATURE_COLUMNS
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info("✓ All required columns present")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        initial_rows = len(df)
        
        # Check for missing values
        missing_counts = df[Config.FEATURE_COLUMNS].isnull().sum()
        
        if missing_counts.sum() > 0:
            self.logger.warning(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
            
            # Forward fill then backward fill
            df[Config.FEATURE_COLUMNS] = df[Config.FEATURE_COLUMNS].fillna(method='ffill').fillna(method='bfill')
            
            # Drop if still missing
            df = df.dropna(subset=Config.FEATURE_COLUMNS)
            
            self.logger.info(f"Rows after handling missing values: {len(df)} (removed {initial_rows - len(df)})")
        else:
            self.logger.info("✓ No missing values detected")
        
        return df
    
    def _quality_checks(self, df: pd.DataFrame):
        """Perform data quality checks"""
        self.logger.info("Performing data quality checks...")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=[Config.DATE_COLUMN]).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate dates")
        
        # Check for anomalies
        for col in Config.FEATURE_COLUMNS:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
            if outliers > 0:
                self.logger.warning(f"{col}: {outliers} potential outliers detected")
        
        # Date range
        date_range = f"{df[Config.DATE_COLUMN].min()} to {df[Config.DATE_COLUMN].max()}"
        self.logger.info(f"Date range: {date_range}")
        
        # Basic statistics
        self.logger.info("\nData Statistics:")
        self.logger.info(f"\n{df[Config.FEATURE_COLUMNS].describe()}")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Advanced feature engineering for financial time series"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features"""
        if not Config.ADD_TECHNICAL_INDICATORS:
            return df
        
        self.logger.info("Creating technical indicators...")
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Moving averages
        df = self._add_moving_averages(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Remove NaN rows created by indicators
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        
        self.logger.info(f"Features created. Rows: {initial_len} -> {len(df)}")
        self.logger.info(f"Total features: {len(df.columns)}")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        df['HL_Range'] = df['High'] - df['Low']
        df['OC_Range'] = abs(df['Close'] - df['Open'])
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_HL'] = (df['High'] - df['Low']) / df['Low']
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators"""
        # Simple Moving Averages
        for period in Config.INDICATORS.get('SMA', []):
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'Close_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in Config.INDICATORS.get('EMA', []):
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            df[f'Close_EMA_{period}_Ratio'] = df['Close'] / df[f'EMA_{period}']
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        # RSI
        for period in Config.INDICATORS.get('RSI', []):
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if Config.INDICATORS.get('MACD'):
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # Bollinger Bands
        if Config.INDICATORS.get('BOLLINGER'):
            period, std_dev = Config.INDICATORS['BOLLINGER']
            df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (std_dev * bb_std)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (std_dev * bb_std)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
        
        # Average True Range
        for period in Config.INDICATORS.get('ATR', []):
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        
        # Rolling volatility
        for period in Config.INDICATORS.get('VOLATILITY', []):
            df[f'Volatility_{period}'] = df['Close'].pct_change().rolling(window=period).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding date)"""
        return [col for col in df.columns if col != Config.DATE_COLUMN]


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """Handle data scaling and sequence preparation"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.scaler = None
        self.feature_columns = None
    
    def fit_scaler(self, data: np.ndarray):
        """Fit scaler on training data"""
        if Config.SCALER_TYPE == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(data)
        self.logger.info(f"✓ {Config.SCALER_TYPE.capitalize()} scaler fitted")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray, feature_idx: int = None) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if feature_idx is not None:
            # Create dummy array for inverse transform
            dummy = np.zeros((data.shape[0], self.scaler.n_features_in_))
            dummy[:, feature_idx] = data.flatten()
            result = self.scaler.inverse_transform(dummy)
            return result[:, feature_idx]
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, filepath: str):
        """Save scaler to disk"""
        joblib.dump(self.scaler, filepath)
        self.logger.info(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load scaler from disk"""
        self.scaler = joblib.load(filepath)
        self.logger.info(f"✓ Scaler loaded from {filepath}")
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        target_idx: int,
        window_size: int,
        prediction_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM"""
        X, y = [], []
        
        for i in range(len(data) - window_size - prediction_horizon + 1):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size:i+window_size+prediction_horizon, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created sequences - X: {X.shape}, y: {y.shape}")
        
        return X, y


# =============================================================================
# MODEL BUILDER
# =============================================================================

class ModelBuilder:
    """Build and compile LSTM model"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def build_model(
        self, 
        input_shape: Tuple[int, int], 
        output_size: int
    ) -> Sequential:
        """Build advanced LSTM model architecture"""
        self.logger.info("Building model architecture...")
        
        model = Sequential(name='XAU_USD_LSTM')
        
        # First LSTM layer
        if Config.USE_BIDIRECTIONAL:
            model.add(Bidirectional(
                LSTM(
                    Config.LSTM_UNITS[0],
                    return_sequences=True,
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.RECURRENT_DROPOUT,
                    kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG)
                ),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                Config.LSTM_UNITS[0],
                return_sequences=True,
                dropout=Config.DROPOUT_RATE,
                recurrent_dropout=Config.RECURRENT_DROPOUT,
                kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG),
                input_shape=input_shape
            ))
        
        if Config.USE_BATCH_NORM:
            model.add(BatchNormalization())
        
        # Additional LSTM layers
        for units in Config.LSTM_UNITS[1:-1]:
            if Config.USE_BIDIRECTIONAL:
                model.add(Bidirectional(
                    LSTM(
                        units,
                        return_sequences=True,
                        dropout=Config.DROPOUT_RATE,
                        recurrent_dropout=Config.RECURRENT_DROPOUT,
                        kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG)
                    )
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=True,
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.RECURRENT_DROPOUT,
                    kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG)
                ))
            
            if Config.USE_BATCH_NORM:
                model.add(BatchNormalization())
        
        # Final LSTM layer
        if Config.USE_BIDIRECTIONAL:
            model.add(Bidirectional(
                LSTM(
                    Config.LSTM_UNITS[-1],
                    dropout=Config.DROPOUT_RATE,
                    recurrent_dropout=Config.RECURRENT_DROPOUT,
                    kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG)
                )
            ))
        else:
            model.add(LSTM(
                Config.LSTM_UNITS[-1],
                dropout=Config.DROPOUT_RATE,
                recurrent_dropout=Config.RECURRENT_DROPOUT,
                kernel_regularizer=l1_l2(l1=Config.L1_REG, l2=Config.L2_REG)
            ))
        
        if Config.USE_BATCH_NORM:
            model.add(BatchNormalization())
        
        model.add(Dropout(Config.DROPOUT_RATE))
        
        # Output layer
        model.add(Dense(output_size))
        
        # Compile model
        optimizer = Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.logger.info("✓ Model built successfully")
        self.logger.info(f"\nModel Summary:")
        model.summary(print_fn=lambda x: self.logger.info(x))
        
        return model
    
    def get_callbacks(self) -> List:
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{{epoch:03d}}_loss_{{val_loss:.4f}}.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=Config.REDUCE_LR_FACTOR,
                patience=Config.REDUCE_LR_PATIENCE,
                min_lr=Config.MIN_LEARNING_RATE,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=os.path.join(Config.LOG_DIR, f'tensorboard_{timestamp}'),
                histogram_freq=1
            ),
            
            # CSV Logger
            CSVLogger(
                os.path.join(Config.LOG_DIR, Config.METRICS_NAME),
                append=True
            )
        ]
        
        return callbacks


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = None
        self.history = None
    
    def train(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: List
    ):
        """Train the model"""
        self.logger.info("="*80)
        self.logger.info("Starting model training...")
        self.logger.info("="*80)
        
        self.model = model
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("✓ Training completed")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model on test set"""
        self.logger.info("Evaluating model on test set...")
        
        test_loss, test_mae, test_mape = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_mape': float(test_mape)
        }
        
        self.logger.info(f"Test Loss: {test_loss:.6f}")
        self.logger.info(f"Test MAE: {test_mae:.6f}")
        self.logger.info(f"Test MAPE: {test_mape:.2f}%")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        self.model.save(filepath)
        self.logger.info(f"✓ Model saved to {filepath}")


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Comprehensive backtesting and evaluation"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def directional_accuracy(
        self,
        model: Sequential,
        X: np.ndarray,
        y_true: np.ndarray,
        preprocessor: DataPreprocessor,
        target_idx: int
    ) -> Dict:
        """Calculate directional prediction accuracy"""
        self.logger.info("="*80)
        self.logger.info("Running Directional Accuracy Analysis...")
        self.logger.info("="*80)
        
        predictions = model.predict(X, verbose=0)
        
        # Track per-horizon accuracy
        horizon_accuracies = []
        total_correct = 0
        total_predictions = 0
        
        for horizon in range(predictions.shape[1]):
            correct = 0
            total = 0
            
            for i in range(len(predictions)):
                if i == 0:
                    continue
                
                # Get actual direction
                actual_direction = np.sign(y_true[i, horizon] - y_true[i-1, horizon])
                
                # Get predicted direction
                pred_direction = np.sign(predictions[i, horizon] - predictions[i-1, horizon])
                
                if actual_direction == pred_direction:
                    correct += 1
                total += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            horizon_accuracies.append(accuracy)
            total_correct += correct
            total_predictions += total
            
            self.logger.info(f"Horizon {horizon+1}: {accuracy:.2f}% ({correct}/{total})")
        
        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        results = {
            'overall_accuracy': overall_accuracy,
            'per_horizon_accuracy': horizon_accuracies,
            'total_correct': total_correct,
            'total_predictions': total_predictions
        }
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"OVERALL DIRECTIONAL ACCURACY: {overall_accuracy:.2f}%")
        self.logger.info(f"{'='*80}\n")
        
        return results
    
    def calculate_regression_metrics(
        self,
        model: Sequential,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Calculate regression performance metrics"""
        predictions = model.predict(X, verbose=0)
        
        mse = np.mean((predictions - y_true) ** 2)
        mae = np.mean(np.abs(predictions - y_true))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
        
        self.logger.info("Regression Metrics:")
        self.logger.info(f"  MSE: {mse:.6f}")
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  RMSE: {rmse:.6f}")
        self.logger.info(f"  R² Score: {r2:.4f}")
        
        return metrics


# =============================================================================
# RESULTS MANAGER
# =============================================================================

class ResultsManager:
    """Save and manage training results"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def save_results(self, results: Dict, filepath: str):
        """Save results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"✓ Results saved to {filepath}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report"""
        report = []
        report.append("="*80)
        report.append("XAU/USD LSTM MODEL - TRAINING REPORT")
        report.append("="*80)
        report.append(f"\nTraining Date: {results.get('timestamp', 'N/A')}")
        report.append(f"Model: {results.get('model_name', 'N/A')}")
        report.append(f"\nDataset Information:")
        report.append(f"  Total Samples: {results.get('total_samples', 'N/A')}")
        report.append(f"  Training Samples: {results.get('train_samples', 'N/A')}")
        report.append(f"  Validation Samples: {results.get('val_samples', 'N/A')}")
        report.append(f"  Test Samples: {results.get('test_samples', 'N/A')}")
        
        report.append(f"\nModel Configuration:")
        report.append(f"  Window Size: {Config.WINDOW_SIZE}")
        report.append(f"  Prediction Horizon: {Config.PREDICTION_HORIZON}")
        report.append(f"  LSTM Units: {Config.LSTM_UNITS}")
        report.append(f"  Dropout Rate: {Config.DROPOUT_RATE}")
        
        if 'test_metrics' in results:
            report.append(f"\nTest Set Performance:")
            for key, value in results['test_metrics'].items():
                report.append(f"  {key}: {value}")
        
        if 'directional_accuracy' in results:
            report.append(f"\nDirectional Accuracy:")
            report.append(f"  Overall: {results['directional_accuracy']['overall_accuracy']:.2f}%")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class XAUUSDPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self):
        self.logger = setup_logging()
        Config.create_directories()
        
        self.data_loader = DataLoader(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
        self.preprocessor = DataPreprocessor(self.logger)
        self.model_builder = ModelBuilder(self.logger)
        self.trainer = ModelTrainer(self.logger)
        self.backtester = Backtester(self.logger)
        self.results_manager = ResultsManager(self.logger)
    
    def run(self):
        """Execute complete training pipeline"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Load data
            df = self.data_loader.load_data(Config.DATA_FILE)
            
            # Step 2: Feature engineering
            df = self.feature_engineer.create_features(df)
            feature_columns = self.feature_engineer.get_feature_columns(df)
            
            # Step 3: Prepare data
            data = df[feature_columns].values
            target_idx = feature_columns.index(Config.TARGET_COLUMN)
            
            # Step 4: Train/test split
            train_size = int(len(data) * Config.TRAIN_TEST_SPLIT)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Step 5: Fit scaler on training data
            self.preprocessor.fit_scaler(train_data)
            train_scaled = self.preprocessor.transform(train_data)
            test_scaled = self.preprocessor.transform(test_data)
            
            # Step 6: Create sequences
            X_train, y_train = self.preprocessor.create_sequences(
                train_scaled, target_idx, Config.WINDOW_SIZE, Config.PREDICTION_HORIZON
            )
            X_test, y_test = self.preprocessor.create_sequences(
                test_scaled, target_idx, Config.WINDOW_SIZE, Config.PREDICTION_HORIZON
            )
            
            # Step 7: Validation split
            val_size = int(len(X_train) * Config.VALIDATION_SPLIT)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]
            
            self.logger.info(f"\nDataset splits:")
            self.logger.info(f"  Train: {len(X_train)} samples")
            self.logger.info(f"  Validation: {len(X_val)} samples")
            self.logger.info(f"  Test: {len(X_test)} samples")
            
            # Step 8: Build model
            model = self.model_builder.build_model(
                input_shape=(Config.WINDOW_SIZE, len(feature_columns)),
                output_size=Config.PREDICTION_HORIZON
            )
            
            # Step 9: Train model
            callbacks = self.model_builder.get_callbacks()
            self.trainer.train(model, X_train, y_train, X_val, y_val, callbacks)
            
            # Step 10: Evaluate
            test_metrics = self.trainer.evaluate(X_test, y_test)
            
            # Step 11: Backtesting
            directional_results = self.backtester.directional_accuracy(
                model, X_test, y_test, self.preprocessor, target_idx
            )
            regression_metrics = self.backtester.calculate_regression_metrics(
                model, X_test, y_test
            )
            
            # Step 12: Save artifacts
            model_path = os.path.join(Config.MODEL_DIR, Config.MODEL_NAME)
            self.trainer.save_model(model_path)
            
            scaler_path = os.path.join(Config.MODEL_DIR, Config.SCALER_NAME)
            self.preprocessor.save_scaler(scaler_path)
            
            config_path = os.path.join(Config.MODEL_DIR, Config.CONFIG_NAME)
            Config.save_config(config_path)
            
            # Step 13: Compile results
            results = {
                'timestamp': timestamp,
                'model_name': Config.MODEL_NAME,
                'total_samples': len(data),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': len(feature_columns),
                'features': feature_columns,
                'test_metrics': test_metrics,
                'directional_accuracy': directional_results,
                'regression_metrics': regression_metrics
            }
            
            results_path = os.path.join(Config.RESULTS_DIR, f'results_{timestamp}.json')
            self.results_manager.save_results(results, results_path)
            
            # Step 14: Generate report
            report = self.results_manager.generate_report(results)
            self.logger.info(f"\n{report}")
            
            report_path = os.path.join(Config.RESULTS_DIR, f'report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info("="*80)
            self.logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("XAU/USD LSTM Production Training System")
    print("="*80 + "\n")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run pipeline
    pipeline = XAUUSDPipeline()
    results = pipeline.run()
    
    print("\n✓ Training complete. Check the 'results' directory for detailed reports.")
    print(f"✓ Model saved in: {Config.MODEL_DIR}")
    print(f"✓ Logs saved in: {Config.LOG_DIR}\n")


if __name__ == "__main__":
    main()
