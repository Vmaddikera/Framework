import warnings
warnings.filterwarnings("ignore")

import math
import os
import sys
import csv
import pickle
import pandas as pd
import numpy as np
from synapse.backtesting import Backtest, Strategy
from synapse.trade_management.tp_sl.atr_tm import ATR_RR_TradeManagement
from synapse.risk_management.equal_weigh_rm import EqualRiskManagement
from synapse.multi_backtester.multi_backtester import MultiBacktest
from synapse.risk_engine.Single_Risk_Engine import RiskEngine

# Set parent directory (adjust as needed)
parent_dir = r"/root/projects/D1-ML/Mega_ML_Framework"
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug prints to verify directory structure
print("Parent dir:", parent_dir)
print("sys.path:", sys.path)
print("Contents of parent dir:", os.listdir(parent_dir))
print("Contents of scalers:", os.listdir(os.path.join(parent_dir, "scalers")))
print("Contents of ML_models:", os.listdir(os.path.join(parent_dir, "ML_models_regression")))
print("Contents of feature_engineering:", os.listdir(os.path.join(parent_dir, "feature_engineering")))
print("Contents of target_engineering:", os.listdir(os.path.join(parent_dir, "target_engineering_regression")))
print("Contents of trade_logic:", os.listdir(os.path.join(parent_dir, "trade_logic_regression")))

# Import components
from feature_engineering import *
from scalers import *
from target_engineering_regression import *
from ML_models_regression import *
from trade_logic_regression import *

CustomFeatureEngineering = SMAVolatilityRetFeatures
CustomScaler = StandardScalerTransformer
CustomTargetEngineering = TargetFutureReturn
CustomModel = SGDRegressorModel
CustomTradeLogic = TradeLogicVariation1

# Thresholds for signals
UPPER_THRESHOLD = 0.0
LOWER_THRESHOLD = 0.0

class CombinedMLStrategy(RiskEngine, CustomTradeLogic):
    # Strategy parameters
    train_percentage = 0.95      # Portion of the available data to consider initially
    initial_percentage = 0.15    # Percentage of the "full" X to use for training
    risk_reward_ratio = 1.25
    atr_period = 7
    atr_multiplier = 5
    initial_risk_per_trade = 0.01
    y_period = 2
    buffer = 100  # Train every 100 ticks

    def init(self):
        super().init()

        # Initialize tick counter
        self.tick_counter = 0

        # Initialize trade & risk management
        self.trade_management_strategy = ATR_RR_TradeManagement(self, risk_reward_ratio=self.risk_reward_ratio, atr_period=self.atr_period, atr_multiplier=self.atr_multiplier)
        self.risk_management_strategy = EqualRiskManagement(self, initial_risk_per_trade=self.initial_risk_per_trade)
        self.total_trades = len(self.closed_trades)

        # Compute features from the original OHLC data using chosen feature module
        feature_dict = CustomFeatureEngineering.apply_features(self.data.df.copy())
        self.feature_names = list(feature_dict.keys())

        # Assign each feature to an indicator using self.I
        for name, values in feature_dict.items():
            setattr(self, name, self.I(lambda v=values: v, name=name))

        # Compute full target series
        self.target = CustomTargetEngineering.get_target(self.data.df, y_period=self.y_period).values

        # Determine training split based on train_percentage and data length
        self.split = int(self.train_percentage * len(self.data.df))

        # Prepare an empty forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')
        
        # Instantiate the Scalar and ML model
        self.scaler = None
        self.ml_model = CustomModel()

        # Logging
        os.makedirs('logs', exist_ok=True)
        cols = ['timestamp'] + self.feature_names + ['actual_target', 'forecast', 'forecast_signal', 'executed_signal', 'signal_correct']
        self._log_file = open('logs/strategy_log.csv', 'w', newline='')
        self._csv = csv.DictWriter(self._log_file, fieldnames=cols)
        self._csv.writeheader()
        os.makedirs('models', exist_ok=True)

    def next(self):
        super().next()

        self.tick_counter += 1

        # Ensure sufficient data for training/prediction
        if len(self.data) <= self.split:
            return

        # train/update on schedule
        if self.tick_counter % self.buffer == 0:
            
            # Compute how many rows to train on (exclude the very last bar)
            train_length = int(self.initial_percentage * (len(self.data) - 1))

            # Build training features (drop the newest bar, then take last train_length)
            X = np.array([getattr(self, fname)[:-1] for fname in self.feature_names]).T[-train_length:]

            # Build corresponding targets
            y = self.target[:len(self.data) - 1][-train_length:]

            # Remove any NaN values from y (and corresponding rows in X)
            isnan = np.isnan(y)
            X = X[~isnan]
            y = y[~isnan]

            if len(X) == 0 or len(y) == 0:
                print("No valid training data available.")
                return

            # Scale training data and train the ML model
            X_scaled, self.scaler = CustomScaler().scale(X)
            self.ml_model.fit(X_scaled, y)

            print(f"Trained model at tick number : {self.tick_counter}")

            # Save model and scaler
            path = f"models/model_tick_{self.tick_counter}.pkl"
            with open(path, 'wb') as f:
                pickle.dump({'model': self.ml_model, 'scaler': self.scaler}, f)

        # skip prediction if not ready
        if self.scaler is None:
            return

        # Build latest feature vector from each feature indicator (for the last bar)
        latest_features = np.array([getattr(self, fname)[-1] for fname in self.feature_names]).reshape(1, -1)
        X_latest_scaled = self.scaler.transform(latest_features)
        forecast = self.ml_model.predict(X_latest_scaled)[0]
        self.forecasts[-1] = forecast

        # Trade logic method
        self.execute_trade_logic(forecast)

        # Logging
        ts = self.data.df.index[len(self.data) - 1]
        row = {'timestamp': ts}
        row.update({f: getattr(self, f)[-1] for f in self.feature_names})
        actual = self.target[len(self.data) - 1]
        row['actual_target'] = actual
        row['forecast'] = forecast
        row['forecast_signal'] = 1 if forecast > UPPER_THRESHOLD else (-1 if forecast < LOWER_THRESHOLD else 0)
        row['executed_signal'] = 1 if actual > UPPER_THRESHOLD else (-1 if actual < LOWER_THRESHOLD else 0)
        row['signal_correct'] = 1 if (row['forecast_signal'] * actual) > 0 else 0
        self._csv.writerow(row)

    def stop(self):
        try:
            self._log_file.close()
        except:
            pass
        super().stop()

if __name__ == '__main__':
    bt = MultiBacktest(CombinedMLStrategy, cash=100000, commission=0.00005, fail_fast=False, look_ahead_bias=False, trade_on_close=True, show_progress=True)
    stats = bt.backtest_stock('AUDUSD', '1hour', 'forex', 'metaquotes')
    print(stats)