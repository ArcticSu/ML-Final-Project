from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class LabelPowerset(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=None):
        """
        Label Powerset实现，将多标签问题转化为多类分类问题。
        参数:
        - base_classifier: 基分类器（默认使用 DecisionTreeClassifier）
        """
        self.base_classifier = base_classifier  # 保存分类器
        self.classifier = None
        self.label_combinations = None

    def fit(self, X, y):
        """
        训练模型。
        参数:
        - X: 特征矩阵
        - y: 标签矩阵（多标签）
        """
        # 将多标签转化为多类问题
        self.label_combinations, y_combined = np.unique(y, axis=0, return_inverse=True)
        self.classifier = self.base_classifier.fit(X, y_combined)

    def predict(self, X):
        """
        使用训练好的模型进行预测。
        参数:
        - X: 特征矩阵
        返回:
        - 多标签预测结果
        """
        # 多类分类预测转化为多标签
        y_combined_pred = self.classifier.predict(X)
        return self.label_combinations[y_combined_pred]

    def predict_proba(self, X):
        """
        返回每个标签组合的概率。
        参数:
        - X: 特征矩阵
        返回:
        - 标签组合的概率
        """
        return self.classifier.predict_proba(X)
