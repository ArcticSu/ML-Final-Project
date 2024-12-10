from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DiSMEC(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        DiSMEC: 分布式稀疏神经网络的简化实现，使用 Logistic Regression 模拟。
        """
        self.models = []

    def fit(self, X, y):
        """
        训练每个标签的独立分类器。
        参数:
        - X: 特征矩阵
        - y: 标签矩阵
        """
        self.models = []
        for i in range(y.shape[1]):
            model = LogisticRegression(solver='liblinear', max_iter=1000)
            model.fit(X, y[:, i])
            self.models.append(model)

    def predict(self, X):
        """
        使用训练好的分类器进行预测。
        参数:
        - X: 特征矩阵
        返回:
        - 多标签预测结果
        """
        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return preds
