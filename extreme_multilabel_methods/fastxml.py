from sklearn.tree import DecisionTreeClassifier
import numpy as np

class FastXML:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """训练多个决策树"""
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        """预测多个树的平均结果"""
        n_classes = len(self.trees[0].classes_)  # 确保获取类别数
        preds = np.zeros((X.shape[0], n_classes))  # 初始化预测矩阵

        for tree in self.trees:
            # 每棵树的预测概率
            tree_preds = tree.predict_proba(X)
            for i, class_probs in enumerate(tree_preds):
                if len(class_probs.shape) == 1:  # 如果是 1D 数组
                    preds[:, i] += class_probs  # 直接累加概率
                elif class_probs.shape[1] > 1:  # 如果是 2D 数组
                    preds[:, i] += class_probs[:, 1]  # 累加标签为 1 的概率

        # 对所有树的预测结果取平均
        preds /= self.n_trees
        return (preds > 0.5).astype(int)
