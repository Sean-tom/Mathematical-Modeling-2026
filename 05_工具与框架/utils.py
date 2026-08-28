#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学建模竞赛通用工具库
提供数据处理、模型评估、可视化等常用函数
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    """数据处理类"""
    
    @staticmethod
    def handle_missing_values(df, method='mean'):
        """处理缺失值
        
        Args:
            df: pandas DataFrame
            method: 处理方法 'mean', 'median', 'forward_fill', 'drop'
        """
        if method == 'mean':
            return df.fillna(df.mean())
        elif method == 'median':
            return df.fillna(df.median())
        elif method == 'forward_fill':
            return df.fillna(method='ffill')
        elif method == 'drop':
            return df.dropna()
        return df
    
    @staticmethod
    def remove_outliers(df, method='iqr', threshold=1.5):
        """移除异常值
        
        Args:
            df: pandas DataFrame
            method: 方法 'iqr' 或 'zscore'
            threshold: 阈值
        """
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return df[(df >= lower_bound) & (df <= upper_bound)].dropna()
        elif method == 'zscore':
            z_scores = np.abs((df - df.mean()) / df.std())
            return df[(z_scores < threshold).all(axis=1)]
        return df
    
    @staticmethod
    def normalize_data(df, method='standard'):
        """数据标准化
        
        Args:
            df: pandas DataFrame
            method: 方法 'standard' 或 'minmax'
        """
        if method == 'standard':
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df


class ModelEvaluator:
    """模型评估类"""
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """回归模型评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    @staticmethod
    def cross_validation_score(model, X, y, cv=5, scoring='r2'):
        """交叉验证评分"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }


class Visualizer:
    """可视化类"""
    
    @staticmethod
    def plot_time_series(df, title='时间序列图', figsize=(12, 5)):
        """绘制时间序列"""
        plt.figure(figsize=figsize)
        plt.plot(df)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('时间')
        plt.ylabel('数值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_distribution(data, title='分布图', figsize=(10, 5)):
        """绘制分布图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'{title} - 直方图')
        axes[1].boxplot(data)
        axes[1].set_title(f'{title} - 箱线图')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(10, 8)):
        """绘制相关性矩阵"""
        plt.figure(figsize=figsize)
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('相关性矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred, title='实际值vs预测值'):
        """绘制实际值vs预测值"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt


class OptimizationHelper:
    """优化算法辅助类"""
    
    @staticmethod
    def normalize_objective(values, method='minmax'):
        """目标函数标准化（用于多目标优化）"""
        if method == 'minmax':
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val == min_val:
                return np.zeros_like(values)
            return (values - min_val) / (max_val - min_val)
        elif method == 'zscore':
            return (values - np.mean(values)) / (np.std(values) + 1e-8)
        return values
    
    @staticmethod
    def weighted_sum(objectives, weights):
        """加权求和法处理多目标优化"""
        return np.sum(np.array(objectives) * np.array(weights))


if __name__ == '__main__':
    print('数学建模竞赛工具库已加载')
