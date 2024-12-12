from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score

def evaluate_multilabel(y_true, y_pred):
    """
    全面评估多标签分类模型性能。

    参数:
    y_true -- 真实标签 (NumPy array)
    y_pred -- 预测标签 (NumPy array)

    返回:
    results -- 包含所有评估指标的字典
    """
    results = {}

    # Subset Accuracy
    results['Subset Accuracy'] = accuracy_score(y_true, y_pred)

    # Hamming Loss
    results['Hamming Loss'] = hamming_loss(y_true, y_pred)

    # Micro Precision, Recall, F1-Score (调整 zero_division 参数避免警告)
    results['Micro Precision'] = precision_score(y_true, y_pred, average='micro', zero_division=1)
    results['Micro Recall'] = recall_score(y_true, y_pred, average='micro', zero_division=1)
    results['Micro F1'] = f1_score(y_true, y_pred, average='micro', zero_division=1)

    # Macro Precision, Recall, F1-Score
    results['Macro Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=1)
    results['Macro Recall'] = recall_score(y_true, y_pred, average='macro', zero_division=1)
    results['Macro F1'] = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # 加权平均指标 (自定义权重)
    micro_f1 = results['Micro F1']
    macro_f1 = results['Macro F1']
    results['Weighted F1'] = 0.7 * micro_f1 + 0.3 * macro_f1

    # Jaccard Index
    results['Jaccard Index (Micro)'] = jaccard_score(y_true, y_pred, average='micro', zero_division=1)
    results['Jaccard Index (Macro)'] = jaccard_score(y_true, y_pred, average='macro', zero_division=1)

    return results
