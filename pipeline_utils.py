# -*- coding: utf-8 -*-
import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.preprocessors.text_preprocessor import TextPreprocessor, PrinterNonTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# scikeras needs scikit-learn version 1.5.2 (default 1.6.1 throws: 'super' object has no attribute '__sklearn_tags__'.
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score
import pandas as pd
from plot_utils import plot_score_distribution, plot_roc_curve, plot_precision_recall_curve


def build_model(meta):
    input_dim = meta['n_features_in_']
    num_output_classes = meta['n_classes_']
    model = keras.Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(num_output_classes,
              activation='softmax' if num_output_classes > 1 else 'sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def print_anomaly_confusion_table(ok_data_len, ok_data_predicted_ok, anomalous_data_len, anomalous_data_predicted_anomalous):
    # Compute confusion matrix values
    TP = anomalous_data_predicted_anomalous
    FN = anomalous_data_len - TP
    TN = ok_data_predicted_ok
    FP = ok_data_len - TN

    # Count table
    print("Confusion Matrix (Counts):")
    print(f"{'is_anomaly/predicted':<20} {'True':>8} {'False':>8}")
    print(f"{'True':<20} {TP:8} {FN:8}")
    print(f"{'False':<20} {FP:8} {TN:8}")

    # Row-wise percentages
    print("\nConfusion Matrix (Row-wise %):")
    print(f"{'is_anomaly/predicted':<20} {'True':>8} {'False':>8}")
    print(f"{'True':<20} {TP/anomalous_data_len:8.2%} {FN/anomalous_data_len:8.2%}")
    print(f"{'False':<20} {FP/ok_data_len:8.2%} {TN/ok_data_len:8.2%}")

    print("\n\n")

    return TP, FP, TN, FN


def create_pipelines(X_cols, estimators=None):
    """
    Creates a pipelines with text preprocessing and the given estimators.
    """
    if not estimators:
        estimators = [
            ("LogRegression", LogisticRegression()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            # ('xgb', XGBClassifier(use_label_encoder=False,
             # eval_metric='logloss', random_state=42)),
            ('lgbm', LGBMClassifier(use_label_encoder=False,
             random_state=42, verbosity=-1)),
            # ('svc', SVC(probability=True, random_state=42)),  # probability=True enables predict_proba; Probably takes the longest time
            # needs scikit-learn version 1.5.2 (default 1.6.1 throws 'super' object has no attribute '__sklearn_tags__'.
            ('KerasNNClassifier', KerasClassifier(
                model=build_model, epochs=5, verbose=0)),
            ('KNeighborsClassifier', KNeighborsClassifier()),
        ]
    voting = VotingClassifier(
        estimators=estimators,
        voting="soft",
    )

    cols = [('BagOfWords' + col, TfidfVectorizer(use_idf=False, norm=None, dtype=np.float64)
, col) for col in X_cols]
    basic_pipeline = [
        ('TextPreprocessor', TextPreprocessor(X_cols)),
        ('VectorizeText', ColumnTransformer(cols)),
    ]
    pipelines = []
    for estimator in estimators + [('VotingClassifier', voting)]:
        pipelines.append(Pipeline(basic_pipeline + [estimator]))
    return pipelines


def create_pipelines_for_probs(X_cols, estimators=None):
    """
    Creates a pipelines with text preprocessing and the given estimators.
    """
    if not estimators:
        estimators = [
            # probability=True enables predict_proba;
            ("LogRegression", LogisticRegression()),
            # ("LogRegressionSagaL1", LogisticRegression(solver='saga', penalty='l1', C=1.0, max_iter=1000)),
            # ("LogRegressionSagaL2", LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('sgdLogLoss', SGDClassifier(loss='log_loss', random_state=42)),
            ('sgdModifiedHuber', SGDClassifier(
                loss='modified_huber', random_state=42)),
            # SCV Probably takes the longest time
            # ('svc', SVC(probability=True, random_state=42)),
            ('ComplementNB', ComplementNB()),
            ('MultinomialNB', MultinomialNB()),
            ('BernoulliNB', BernoulliNB()),
            ('KNeighborsClassifier', KNeighborsClassifier()),
            # ('QuadraticDiscriminantAnalysis',QuadraticDiscriminantAnalysis()), # Does not make sense in high dim sparse data
        ]
    voting = VotingClassifier(
        estimators=estimators,
        voting="soft",
    )

    cols = [('BagOfWords' + col, TfidfVectorizer(use_idf=False, norm=None, dtype=np.float64), col) for col in X_cols]
    basic_pipeline = [
        ('TextPreprocessor', TextPreprocessor(X_cols)),
        ('VectorizeText', ColumnTransformer(cols)),
    ]
    pipelines = []
    for estimator in estimators + [('VotingClassifier', voting)]:
        pipelines.append(Pipeline(basic_pipeline + [estimator]))
    return pipelines


def evaluate_pipeline_from_probs(pipe, X_train, y_train, X_test, y_test, threshold=0.6, y_anomalous=None, target_col=None):
    """
    Fits the pipeline, predicts, and evaluates accuracy and timing.
    """
    # Time the fit
    start_fit = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Time the predict
    start_pred = time.time()
    probs = pipe.predict_proba(X_test)
    pred_time = time.time() - start_pred

    # print("Classifier knows classes:", pipe.classes_)
    # print("y_test unique classes:", np.unique(y_test))
    # true_class_probs = probs[np.arange(len(y_test)), y_test]
    # preds = true_class_probs < threshold
    # Above code does not handle classes not seen in data
    # Code belowe does:
    class_indices = {label: i for i, label in enumerate(pipe.classes_)}
    true_class_probs = np.array([
        probs[i, class_indices[y]] if y in class_indices else 0.0
        for i, y in enumerate(y_test)
    ])
    preds = true_class_probs <= threshold

    ok_data_len = len(y_test)
    ok_data_predicted_ok = len(y_test) - sum(preds)

    if not y_anomalous:
        labels = y_test.unique()
        mapping = create_shuffled_mapping(labels)
        y_anomalous = y_test.map(mapping)

    class_indices = {label: i for i, label in enumerate(pipe.classes_)}
    true_class_probs_anomaly = np.array([
        probs[i, class_indices[y]] if y in class_indices else 0.0
        for i, y in enumerate(y_anomalous)
    ])
    preds = true_class_probs_anomaly <= threshold

    scores = np.concatenate([true_class_probs, true_class_probs_anomaly])
    y_true = np.concatenate([np.ones(len(true_class_probs)), np.zeros(len(true_class_probs_anomaly))])

    print("HAS NaN:", np.isnan(scores).any())
    print("ALL FINITE:", np.isfinite(scores).all())
    try:
        plot_roc_curve(y_true, scores)
    except:
        print("Failed ROC Curve")
    try:
        plot_precision_recall_curve(y_true, scores)
    except:
        print("Failed Precision Recall Curve")
    try:
        plot_score_distribution(scores, bins=50)
    except:
        print("Failed Anomaly Score Curve")
    
    anomalous_data_len = len(y_anomalous)
    anomalous_data_predicted_anomalous = sum(preds)

    print("Estimator:", pipe.steps[-1][0])
    print(f"Fit time: {fit_time:.3f}s | Predict time: {pred_time:.3f}s")
    TP, FP, TN, FN = print_anomaly_confusion_table(
        ok_data_len, ok_data_predicted_ok, anomalous_data_len, anomalous_data_predicted_anomalous)
    return TP, FP, TN, FN


def create_shuffled_mapping(categories):
    """
    Creates a shuffled mapping for anomaly detection.
    """
    shuffled = categories.copy()
    if len(categories) <= 1:
        return dict(zip(categories, categories))
    while True:
        np.random.shuffle(shuffled)
        if not np.any(shuffled == categories):
            break
    return dict(zip(categories, shuffled))


def evaluate_pipeline(pipe, X_train, y_train, X_test, y_test, y_anomalous=None, verbose=False):
    """
    Fits the pipeline, predicts, and evaluates accuracy and timing.
    """
    # Time the fit
    start_fit = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Time the predict
    start_pred = time.time()
    y_pred = pipe.predict(X_test)
    pred_time = time.time() - start_pred

    if not y_anomalous:
        labels = y_test.unique()
        mapping = create_shuffled_mapping(labels)
        y_anomalous = y_test.map(mapping)

    ok_data_len = anomalous_data_len = len(y_test)
    y_ok_equal = (np.array(y_test) == np.array(y_pred))
    ok_data_predicted_ok = sum(y_ok_equal)
    y_anomalous_not_equal = (np.array(y_pred) != np.array(y_anomalous))
    anomalous_data_predicted_anomalous = sum(y_anomalous_not_equal)

    print("Estimator:", pipe.steps[-1][0])
    print(f"Fit time: {fit_time:.3f}s | Predict time: {pred_time:.3f}s")
    TP, FP, TN, FN = print_anomaly_confusion_table(
        ok_data_len, ok_data_predicted_ok, anomalous_data_len, anomalous_data_predicted_anomalous)
    return TP, FP, TN, FN
    # Accuracy
    # print("Estimator:", pipe.steps[-1][0])
    acc = accuracy_score(y_test, y_pred)
    # print(f"OK data predicted as ok: {acc:.3f}")
    if y_anomalous is not None:
        acc_anomalous = accuracy_score(y_pred, y_anomalous)
    #     print(f"Anomalous data predicted as ok: {acc_anomalous:.3f}")
    # print(f"Fit time: {fit_time:.3f}s | Predict time: {pred_time:.3f}s")

    # Show some examples
    correct = []
    incorrect = []
    if isinstance(pipe.steps[-1][1], VotingClassifier) and verbose:
        for i in range(len(y_test[:10000])):
            true = y_test.iloc[i]
            pred = y_pred[i]
            if pred == true:
                correct.append((i, true, pred))
            else:
                incorrect.append((i, true, pred))

        print("\nExamples of correct predictions:")
        for i, true, pred in correct[:5]:
            print(f"{X_test.iloc[i]}: True = {true}, Predicted = {pred}")

        print("\nExamples of incorrect predictions:")
        for i, true, pred in incorrect[:5]:
            print(f"{X_test.iloc[i]}: True = {true}, Predicted = {pred}")


def evaluate_pipeline_oodd(pipe, X_train, X_test, X_anomalous=None, target_col=None, type='continuous'):
    """
    Fits the pipeline, predicts, and evaluates accuracy and timing.
    """
    # Time the fit
    start_fit = time.time()
    pipe.fit(X_train)
    fit_time = time.time() - start_fit

    # Time the predict
    start_pred = time.time()
    preds = pipe.predict(X_test)
    pred_time = time.time() - start_pred

    ok_data_len = len(preds)
    ok_data_predicted_ok = len(preds) - sum(preds)

    if not X_anomalous and target_col:
        X_anomalous = X_test.copy()
        if type == 'categorical':
            labels = X_anomalous[target_col].unique()
            print(labels)
            mapping = create_shuffled_mapping(labels)
            X_anomalous[target_col] = X_anomalous[target_col].map(mapping)
        elif type == 'categorical3':
            X_anomalous[target_col] = (X_anomalous[target_col] + 1) % 3 + 1
        elif type == 'categorical2':
            X_anomalous[target_col] = (X_anomalous[target_col] + 1) % 2
        elif type == 'continuous':
            X_anomalous[target_col] = X_anomalous[target_col].astype(int) + \
                np.random.choice([20, -20], size=len(X_anomalous))
        else:
            print("Testing Unexpected type of problem")

        # Time the predict
    start_pred = time.time()
    preds = pipe.predict(X_anomalous)
    pred_time = time.time() - start_pred

    anomalous_data_len = len(preds)
    anomalous_data_predicted_anomalous = sum(preds)

    # print("Estimator:", pipe.steps[-1][0])
    print(f"Fit time: {fit_time:.3f}s | Predict time: {pred_time:.3f}s")
    TP, FP, TN, FN = print_anomaly_confusion_table(
        ok_data_len, ok_data_predicted_ok, anomalous_data_len, anomalous_data_predicted_anomalous)
    return TP, FP, TN, FN
