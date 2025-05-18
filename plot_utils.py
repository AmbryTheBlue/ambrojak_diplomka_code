import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def plot_target_distribution(df: pd.DataFrame, target_col: str, name:str = None ): 
    """
    Plots countplot for categorical.
    """
    if name is None or name=="":
        name = target_col
    counts = df[target_col].value_counts(dropna=False, normalize=True) * 100
    counts.name = name
    categories = counts.index
    categories.name = name
    percents = counts.values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=categories, y=percents, ax=ax, order=categories, edgecolor='black')
    ax.set_ylabel('Percent (%)')
    ax.set_title(f'Distribution of {name} (Percent)')
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    for i, v in enumerate(percents):
        ax.text(i, v + max(percents) * 0.01, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("plots/Distribution-"+name+".pdf")
    plt.show()

def plot_score_distribution_absolute(scores, bins=50):
    """
    Plots histogram of anomaly scores.
    
    Parameters:
    - scores: array-like, anomaly scores
    - bins: int, number of bins for histogram
    """
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=bins)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    # plt.savefig("plots/AnomalyScoresAbsolute.pdf")
    plt.show()

def plot_score_distribution(scores, bins=50):
    """
    Plots histogram of anomaly scores in percentages.
    
    Parameters:
    - scores: array-like, anomaly scores
    - bins: int, number of bins for histogram
    """
    scores = np.asarray(scores)
    # weight each entry so that sum(weights) == 100
    weights = np.ones_like(scores) / len(scores) * 100

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=bins, weights=weights, edgecolor='black')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Percentage')
    plt.title('Distribution of Anomaly Scores')
    plt.tight_layout()
    plt.savefig("plots/AnomalyScores.pdf")
    plt.show()


def plot_pca_anomaly_scatter(X, anomaly_labels, n_components=2):
    """
    Projects data into PCA space and highlights anomalies.
    
    Parameters:
    - X: array-like, feature matrix (n_samples, n_features)
    - anomaly_labels: array-like of 0 (normal) or 1 (anomaly)
    - n_components: int, number of PCA components to project onto
    """
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    normal = anomaly_labels == 0
    anomalous = anomaly_labels == 1
    plt.scatter(X_proj[normal, 0], X_proj[normal, 1], alpha=0.5, label='Normal')
    plt.scatter(X_proj[anomalous, 0], X_proj[anomalous, 1], color='red', label='Anomaly')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA Projection with Anomalies Highlighted')
    plt.show()

def plot_roc_curve(y_true, scores):
    """
    Plots ROC curve for anomaly detection.
    
    Parameters:
    - y_true: array-like of true binary labels (0 normal, 1 anomaly)
    - scores: array-like of anomaly scores
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("plots/ROCcurve.pdf")
    plt.show()

def plot_precision_recall_curve(y_true, scores):
    """
    Plots Precision-Recall curve for anomaly detection.
    
    Parameters:
    - y_true: array-like of true binary labels (0 normal, 1 anomaly)
    - scores: array-like of anomaly scores
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend
    plt.savefig("plots/Precision-Recall.pdf")
    plt.show()

def compare_models_from_components(components_dict):
    """
    components_dict: dict of model name -> [TP, FP, TN, FN]
    """
    metrics = {}
    percents = {}

    for name, (tp, fp, tn, fn) in components_dict.items():
        total = tp + fp + tn + fn
        perc = [100 * x / total for x in (tp, fp, tn, fn)]
        percents[name] = perc

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics[name] = [prec, rec, f1]

    # === Plot confusion components as percent ===
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['TP', 'FP', 'TN', 'FN']
    x = np.arange(len(labels))
    width = 0.2

    for idx, (model, vals) in enumerate(percents.items()):
        ax.bar(x + idx * width, vals, width, label=model)
        for i, v in enumerate(vals):
            ax.text(x[i] + idx * width, v + 1, f'{v:.1f}%', ha='center', va='bottom')

    ax.set_ylabel('Percent (%)')
    ax.set_title('Confusion Matrix Components (Percentage)')
    ax.set_xticks(x + width * (len(percents) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/ConfusionMatrix.pdf")
    plt.show()

    # === Plot precision, recall, f1 ===
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['Precision', 'Recall', 'F1']
    x = np.arange(len(labels))

    for idx, (model, vals) in enumerate(metrics.items()):
        ax.bar(x + idx * width, vals, width, label=model)
        for i, v in enumerate(vals):
            ax.text(x[i] + idx * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Metrics')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/Metrics.pdf")
    plt.show()

    return metrics

def print_latex_table(results: dict, name: str) -> dict:
    """
    Given a dict mapping model names to [TP, FP, TN, FN],
    computes Accuracy, Recall, Precision, F1;
    prints a TeX table of metrics;
    returns a dict with all metrics per model.
    """
    # 1) Compute metrics and print TeX table
    metrics = {}
    for model, (tp, fp, tn, fn) in results.items():
        total    = tp + fp + tn + fn
        accuracy  = (tp + tn) / total       if total else 0
        recall    = tp / (tp + fn)          if (tp + fn) else 0
        precision = tp / (tp + fp)          if (tp + fp) else 0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        # store
        metrics[model] = {
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1': f1
        }

    print(r"\midinsert \clabel[table" + name.replace(' ','') + "]{Model Comparison " + name +"}")
    print(r"\ctable{l|rrrrrrr|r}{")
    print(r"Model & TP & FP & TN & FN & Accuracy & Recall & Precision & F1 \crli \tskip4pt")
    # sort models by F1
    for model, vals in sorted(metrics.items(), key=lambda item: item[1]['F1'], reverse=True):
        tp = vals['TP']; fp = vals['FP']; tn = vals['TN']; fn = vals['FN']
        
        total = tp+fp+tn+fn
        print(f"{model} & {tp/total:.3f} & {fp/total:.3f} & {tn/total:.3f} & {fn/total:.3f} "
              f"& {vals['Accuracy']:.3f} & {vals['Recall']:.3f} & {vals['Precision']:.3f} & {vals['F1']:.3f} \\cr")
    print(r"}")
    print(r"\caption/t Performance metrics per model in " + name + " (sorted by F1)")
    print(r"\endinsert")

    return metrics
