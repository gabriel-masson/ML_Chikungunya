from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import matplotlib.pyplot as plt


class Utils:
    def __init__(self):
        self.fontsize_xy = 14
        self.fontsize_title = 16

    def oi(self):
        print("Oi, Utils!")
        return "xxx!"

    def train_test_balance(self, df):
        X = df.drop('Novo critério (0-3/4-10)', axis=1)
        y = df['Novo critério (0-3/4-10)']

        under_sampler = RandomUnderSampler(
            sampling_strategy={1: 173},
            random_state=42
        )

        X_resampled, y_resampled = under_sampler.fit_resample(X, y)

        over_sampler = RandomOverSampler(
            sampling_strategy={0: 173},
            random_state=42
        )

        X_resampled, y_resampled = over_sampler.fit_resample(
            X_resampled, y_resampled
        )

        return train_test_split(
            X_resampled, y_resampled,
            test_size=0.3,
            random_state=42
        )

    def avaliation_model(self, model, X_test, y_test):
        model_name = model.__class__.__name__
        print(f"\nEvaluating model: {model_name}")
        # =========================
        # Predições
        # =========================
        y_pred = model.predict(X_test)

        # Probabilidade para AUC (se existir)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = roc_auc_score(y_test, y_pred)

        # =========================
        # Métricas principais
        # =========================
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # =========================
        # Especificidade (manual)
        # =========================
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        # =========================
        # Converter para %
        # =========================
        metrics = {
            'accuracy': round(accuracy * 100, 4),
            'recall': round(recall * 100, 4),
            'specificity': round(specificity * 100, 4),
            'precision': round(precision * 100, 4),
            'f1_score': round(f1 * 100, 4),
            'AUC': round(auc * 100, 4)
        }

        # =========================
        # Print organizado
        # =========================
        print("\n===== Model Evaluation =====")
        for k, v in metrics.items():
            print(f"{k}: {v}%")

        return metrics
