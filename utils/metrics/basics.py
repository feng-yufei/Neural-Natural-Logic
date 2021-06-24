from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    return {"acc": accuracy_score(y_true=labels, y_pred=preds)}

def acc_and_f1(preds, labels, id2label_dict):
    result = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)
    result['f1'] = f1
    for id in id2label_dict.keys():
        result["f1_" + id2label_dict[id]] = f1[id]
    return result

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
}