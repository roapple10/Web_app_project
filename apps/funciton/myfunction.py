import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_score(y_pred_list, y_test, average=None, plot=True, axis=0, cmap='Blues'):
    model_name = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []

    for name, y_pred in y_pred_list.items:
        model_name.append(name)
        if average is not None:
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average=average))
            recall.append(recall_score(y_test, y_pred, average=average))
            f1.append(f1_score(y_test, y_pred, average=average))
            roc_auc.append(roc_auc_score(y_test, y_pred, average=average))

            score_list = {
                'model': model_name,
                'accuracy': accuracy,
                f'{average}_avg_precision': precision,
                f'{average}_avg_recall': recall,
                f'{average}_avg_f1_score': f1,
                'roc_auc': roc_auc
            }
        else:
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            roc_auc.append(roc_auc_score(y_test, y_pred))

            score_list = {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }

    score_df = pd.DataFrame(score_list).set_index('model')

    if plot:
        print(score_df.style.background_gradient(axis=axis, cmap=cmap))

    return score_df


def model_list():
    model_list_ = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=1),
        'Ridge Classifier': RidgeClassifier(random_state=1),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC(random_state=1),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=1),
        'Decision Tree': DecisionTreeClassifier(random_state=1),
        'Random Forest': RandomForestClassifier(random_state=1),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=1),
        'AdaBoost Classifier': AdaBoostClassifier(random_state=1),
        'CatBoost Classifier': CatBoostClassifier(random_state=1, verbose=False),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=1),
        'XGBoost': XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=1),
    }
    return model_list_
