import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

# Ensure local module path
parent_dir = os.path.dirname(__file__)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class RegressionEvaluator:
    def __init__(self, y_true=None, y_pred=None, log_csv='logs/strategy_log.csv'):
        if y_true is None or y_pred is None:
            log_path = log_csv
            if not os.path.isabs(log_path):
                script_dir = os.path.dirname(__file__)
                log_path = os.path.join(script_dir, log_csv)
            if not os.path.exists(log_path):
                raise FileNotFoundError(
                    f"Log file not found: {log_path}. "
                    "Please provide y_true and y_pred arrays or correct the log_csv path."
                )
            df = pd.read_csv(log_path, index_col='timestamp', parse_dates=True)
            y_true = df['actual_target']
            y_pred = df['forecast']
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        mask = ~np.isnan(self.y_true) & ~np.isnan(self.y_pred)
        self.y_true = self.y_true[mask]
        self.y_pred = self.y_pred[mask]
        self.metrics = {}

    def compute_metrics(self):
        """
        Compute core sklearn regression metrics.
        """
        self.metrics = {
            'ExplainedVar': explained_variance_score(self.y_true, self.y_pred),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'MedAE': median_absolute_error(self.y_true, self.y_pred),
            'R2': r2_score(self.y_true, self.y_pred)
        }
        return self.metrics

    def plot(self):
        self._plot_overlay()
        self._plot_kde()
        self._plot_box()
        self._plot_violin()
        self._plot_qq()
        self._plot_residuals()
        self._plot_scatter()

    def _plot_overlay(self):
        plt.figure()
        plt.plot(self.y_true, label='Actual')
        plt.plot(self.y_pred, label='Forecast', alpha=0.7)
        plt.legend()
        plt.title('Actual vs Forecast')
        plt.show()

    def _plot_kde(self):
        plt.figure()
        sns.kdeplot(self.y_true, label='Actual')
        plt.title('KDE: Actual')
        plt.show()
        plt.figure()
        sns.kdeplot(self.y_pred, label='Forecast')
        plt.title('KDE: Forecast')
        plt.show()

    def _plot_box(self):
        plt.figure()
        plt.boxplot([self.y_true, self.y_pred], labels=['Actual', 'Forecast'])
        plt.title('Boxplot: Actual and Forecast')
        plt.show()

    def _plot_violin(self):
        plt.figure()
        sns.violinplot(data=[self.y_true, self.y_pred])
        plt.xticks([0, 1], ['Actual', 'Forecast'])
        plt.title('Violin Plot: Actual and Forecast')
        plt.show()

    def _plot_qq(self):
        plt.figure()
        stats.probplot(self.y_true - self.y_pred, dist='norm', plot=plt)
        plt.title('Q-Q Plot Forecast Errors')
        plt.show()

    def _plot_residuals(self):
        residuals = self.y_true - self.y_pred
        plt.figure()
        plt.scatter(self.y_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--')
        plt.xlabel('Forecast')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Forecast')
        plt.show()

    def _plot_scatter(self):
        plt.figure()
        plt.scatter(self.y_true, self.y_pred, alpha=0.6)
        m = max(self.y_true.max(), self.y_pred.max())
        plt.plot([0, m], [0, m], '--')
        plt.xlabel('Actual')
        plt.ylabel('Forecast')
        plt.title('Scatter: Actual vs Forecast')
        plt.show()

if __name__ == '__main__':
    evaluator = RegressionEvaluator()
    metrics = evaluator.compute_metrics()
    print('Regression Metrics:')
    for name, val in metrics.items():
        print(f'{name}: {val:.4f}')

    print('Generating diagnostic plots...')
    evaluator.plot()