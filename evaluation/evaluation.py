from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
import numpy as np


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

    # Micro Precision, Recall, F1-Score
    results['Micro Precision'] = precision_score(y_true, y_pred, average='micro')
    results['Micro Recall'] = recall_score(y_true, y_pred, average='micro')
    results['Micro F1'] = f1_score(y_true, y_pred, average='micro')

    # Macro Precision, Recall, F1-Score
    results['Macro Precision'] = precision_score(y_true, y_pred, average='macro')
    results['Macro Recall'] = recall_score(y_true, y_pred, average='macro')
    results['Macro F1'] = f1_score(y_true, y_pred, average='macro')

    # Jaccard Index
    results['Jaccard Index (Micro)'] = jaccard_score(y_true, y_pred, average='micro')
    results['Jaccard Index (Macro)'] = jaccard_score(y_true, y_pred, average='macro')

    return results



def evaluate_multilabel_custom(y_true, y_pred):
    """
    针对多标签分类任务的改进评估方法。

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

    # Micro Precision, Recall, F1-Score
    results['Micro Precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    results['Micro Recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    results['Micro F1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Macro Precision, Recall, F1-Score
    results['Macro Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['Macro Recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['Macro F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Jaccard Index
    results['Jaccard Index (Micro)'] = jaccard_score(y_true, y_pred, average='micro')
    results['Jaccard Index (Macro)'] = jaccard_score(y_true, y_pred, average='macro')

    # Partial Match Ratio (PMR)
    pmr_list = []
    label_coverage_list = []
    for i in range(y_true.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0])
        pred_set = set(np.where(y_pred[i] == 1)[0])
        
        # 计算 PMR
        if len(pred_set) > 0:
            pmr = len(true_set & pred_set) / len(pred_set)
        else:
            pmr = 0
        pmr_list.append(pmr)
        
        # 计算 Label Coverage
        if len(true_set) > 0:
            coverage = len(true_set & pred_set) / len(true_set)
        else:
            coverage = 0
        label_coverage_list.append(coverage)
    
    results['Partial Match Ratio (PMR)'] = np.mean(pmr_list)
    results['Label Coverage'] = np.mean(label_coverage_list)

    # Weighted Coverage
    results['Weighted Coverage'] = 0.5 * (results['Partial Match Ratio (PMR)'] + results['Label Coverage'])

    return results


def evaluate_multilabel_relaxed(y_true, y_pred):
    """
    针对多标签分类任务的宽松评估方法。

    参数:
    y_true -- 真实标签 (NumPy array)
    y_pred -- 预测标签 (NumPy array)

    返回:
    results -- 包含宽松评估指标的字典
    """
    results = {}

    # Hamming Loss
    results['Hamming Loss'] = hamming_loss(y_true, y_pred)

    # Jaccard Index
    results['Jaccard Index (Micro)'] = jaccard_score(y_true, y_pred, average='micro')
    results['Jaccard Index (Macro)'] = jaccard_score(y_true, y_pred, average='macro')

    # Relaxed Accuracy
    relaxed_accuracy_count = 0
    for i in range(y_true.shape[0]):
        if len(set(np.where(y_true[i] == 1)[0]) & set(np.where(y_pred[i] == 1)[0])) > 0:
            relaxed_accuracy_count += 1
    results['Relaxed Accuracy'] = relaxed_accuracy_count / y_true.shape[0]

    # Soft Precision, Soft Recall, Soft F1
    soft_precision_list = []
    soft_recall_list = []

    for i in range(y_true.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0])
        pred_set = set(np.where(y_pred[i] == 1)[0])

        if len(pred_set) > 0:
            soft_precision = len(true_set & pred_set) / len(pred_set)
        else:
            soft_precision = 0

        if len(true_set) > 0:
            soft_recall = len(true_set & pred_set) / len(true_set)
        else:
            soft_recall = 0

        soft_precision_list.append(soft_precision)
        soft_recall_list.append(soft_recall)

    # 平均 Soft Precision 和 Soft Recall
    results['Soft Precision'] = np.mean(soft_precision_list)
    results['Soft Recall'] = np.mean(soft_recall_list)

    if results['Soft Precision'] + results['Soft Recall'] > 0:
        results['Soft F1'] = 2 * (results['Soft Precision'] * results['Soft Recall']) / (results['Soft Precision'] + results['Soft Recall'])
    else:
        results['Soft F1'] = 0

    return results
