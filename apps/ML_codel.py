######################
# Import libraries
######################
import pickle

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# association between attributes

# from dython.nominal import associations

# CamelCase to snake_case format
#import inflection

# viz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

plt.style.use('fivethirtyeight')
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 1.5})
# change the maximum width in characters of a column (default: 50)
pd.set_option('display.max_colwidth', None)
# change the display precision for better viz
pd.set_option('display.precision', 3)

# encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# oversampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN

# train test split
from sklearn.model_selection import train_test_split

# model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

# model evaluation & tuning hyperparameter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# explainable AI
# import shap


######################
# Page Title
######################


######################
# Input Text Box
######################

def Model_create():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df.drop({'EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'}, axis=1, inplace=True)
    # Reassign target
    df['Attrition'] = df['Attrition'].map({"No": 0, "Yes": 1})

    # categorical
    column_categorical = df.select_dtypes(include=['object']).columns.tolist()
    variation_categorical = dict()

    for col in column_categorical:
        tmp = df[col].unique().tolist()
        tmp.sort()
        variation_categorical[col] = ' ,'.join(str(item) for item in tmp)

    tmp = pd.Series(variation_categorical)
    data_variation_categorical = pd.DataFrame(tmp).T.rename({0: 'data variation'})

    # numerical
    column_numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    variation_numerical = dict()
    for col in column_numerical:
        tmp = f'{df[col].min()} - {df[col].max()}'
        variation_numerical[col] = tmp
    tmp = pd.Series(variation_numerical)
    data_variation_numerical = pd.DataFrame(tmp).T.rename({0: 'data variation'})
    data_variation = pd.concat([data_variation_numerical.rename({'data variation': 'range'}),
                                data_variation_categorical.rename({'data variation': 'variation'})], axis=1).fillna('-')
    data_viz = df.copy()
    data_viz.loc[:, 'Attrition'] = data_viz.loc[:, 'Attrition'].apply(lambda x: 'Attrition' if x == 1 else 'retain')

    columns = data_viz['Attrition']

    attr_crosstab = pd.DataFrame()

    for col in column_categorical:  # column_categorical
        # create crosstab for each attribute
        index = data_viz[col]
        ct = pd.crosstab(index=index, columns=columns, normalize='index', colnames=[None]).reset_index()

        # add prefix to each category
        # format: column name (category)
        col_titleize = inflection.titleize(col)
        ct[col] = ct[col].apply(lambda x: f'{col_titleize} ({x})')

        # rename the column
        ct.rename(columns={col: 'attribute'}, inplace=True)

        # create a single dataframe
        attr_crosstab = pd.concat([attr_crosstab, ct])

    attr_crosstab = attr_crosstab.sort_values('Attrition', ascending=False).reset_index(drop=True)
    data_X = df.drop('Attrition', axis=1)
    data_y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3,
                                                        random_state=1, stratify=data_y)

    ######################
    # Label Encoding
    # I use it for the target variable (label).
    ######################

    le = LabelEncoder()

    le.fit(y_train)
    y_train_encode = le.transform(y_train)
    y_test_encode = le.transform(y_test)

    # drop 1 category if the feature only has 2 categories
    ohe = OneHotEncoder(sparse=False, drop='if_binary')
    ohe.fit(X_train[column_categorical])
    X_train_ohe = ohe.transform(X_train[column_categorical])
    X_test_ohe = ohe.transform(X_test[column_categorical])
    column_ohe = ohe.get_feature_names_out()

    # create dataframe from one-hot encoded features
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=column_ohe, index=X_train.index)
    # combine the numerical and encoded features
    X_train_encode = pd.concat([X_train.drop(columns=column_categorical), X_train_ohe_df], axis=1)
    # create dataframe from one-hot encoded features
    X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=column_ohe, index=X_test.index)
    # combine the numerical and encoded features
    X_test_encode = pd.concat([X_test.drop(columns=column_categorical), X_test_ohe_df], axis=1)
    # combine the X-train and X-test
    data_encode = pd.concat([X_train_encode, X_test_encode], axis=0)
    # combine with the y-train
    data_encode = data_encode.join(pd.Series(y_train_encode, name='Attrition', index=X_train_encode.index),
                                   lsuffix='_1', rsuffix='_2')
    # combine with the y-test
    data_encode = data_encode.join(pd.Series(y_test_encode, name='Attrition', index=X_test_encode.index), lsuffix='_1',
                                   rsuffix='_2')
    # merging the y-train and y-test column
    data_encode['Attrition_1'].fillna(data_encode['Attrition_2'], inplace=True)
    data_encode.drop(columns='Attrition_2', inplace=True)
    data_encode.rename(columns={'Attrition_1': 'Attrition'}, inplace=True)

    # numerical
    df1 = df
    df1.drop('Attrition', axis=1, inplace=True)
    column_numerical1 = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_train_scale = X_train_encode.copy()
    X_test_scale = X_test_encode.copy()

    for i in column_numerical1:
        scaler = MinMaxScaler()
        scaler.fit(X_train_scale[[i]])

        X_train_scale[[i]] = scaler.transform(X_train_scale[[i]])
        X_test_scale[[i]] = scaler.transform(X_test_scale[[i]])

    # combine the X-train and X-test
    data_scale = pd.concat([X_train_scale, X_test_scale], axis=0)
    # combine with the y-train
    data_scale = data_scale.join(pd.Series(y_train_encode, name='Attrition', index=X_train_scale.index), lsuffix='_1',
                                 rsuffix='_2')
    # combine with the y-test
    data_scale = data_scale.join(pd.Series(y_test_encode, name='Attrition', index=X_test_scale.index), lsuffix='_1',
                                 rsuffix='_2')
    # merging the y-train and y-test column
    data_scale['Attrition_1'].fillna(data_scale['Attrition_2'], inplace=True)
    data_scale.drop(columns='Attrition_2', inplace=True)
    data_scale.rename(columns={'Attrition_1': 'Attrition'}, inplace=True)

    data_scale_train = pd.concat(
        [X_train_scale, pd.Series(y_train_encode, name='Attrition', index=X_train_scale.index)], axis=1)

    corr_matrix = data_scale.corr().round(3)
    corr_target = corr_matrix['Attrition'].drop('Attrition')

    smote = SMOTE(random_state=1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scale, y_train_encode)
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_train_smote.columns)
    y_train_smote_df = pd.DataFrame(y_train_smote, columns=['Attrition'])
    data_smote = pd.concat([X_train_smote_df, y_train_smote_df], axis=1)
    numeric_cols1 = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    column_numerical2 = ['Age',
                         'DailyRate',
                         'DistanceFromHome', ]
    X_train_model = X_train_smote.copy()
    y_train_model = y_train_smote.copy()

    X_test_model = X_test_scale.copy()
    y_test_model = y_test_encode.copy()

    model_list = {
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
    # Model1
    # y_pred_list = dict()
    # for name, model in model_list.items():
    #     model.fit(X_train_model, y_train_model)
    #     y_pred_list[name] = model.predict(X_test_model)

    # model_list = {
    #     'Random Forest':RandomForestClassifier(random_state=1),
    #     'Gradient Boosting Classifier':GradientBoostingClassifier(random_state=1),
    #     'AdaBoost Classifier':AdaBoostClassifier(random_state=1),
    #     'CatBoost Classifier':CatBoostClassifier(random_state=1, verbose=False),
    #     'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=1),
    #     'XGBoost':XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'),
    #     'LightGBM':LGBMClassifier(random_state=1),
    # }
    #
    # y_pred_list = dict()
    # # Model2
    # threshold = 13
    # filter = SelectKBest(score_func=f_classif, k=threshold)
    # filter.fit(X_train_model, y_train_model)
    # X_train_filter = filter.transform(X_train_model)
    # X_test_filter = filter.transform(X_test_model)
    # for name, model in model_list.items():
    #     model.fit(X_train_filter, y_train_model)
    #     y_pred_list[name] = model.predict(X_test_filter)
    # Saving the model
    # import pickle
    # pickle.dump(y_pred_list,
    #            open(f'C:/Users/Raylin/PycharmProjects/My_web_app/multi-page-app-main/apps/score_compare.pkl', 'wb'))


def Model_create1():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df.drop({'EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'}, axis=1, inplace=True)
    # Reassign target
    df['Attrition'] = df['Attrition'].map({"No": 0, "Yes": 1})

    # categorical
    column_categorical = df.select_dtypes(include=['object']).columns.tolist()
    variation_categorical = dict()

    for col in column_categorical:
        tmp = df[col].unique().tolist()
        tmp.sort()
        variation_categorical[col] = ' ,'.join(str(item) for item in tmp)

    tmp = pd.Series(variation_categorical)
    data_variation_categorical = pd.DataFrame(tmp).T.rename({0: 'data variation'})

    # numerical
    column_numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    variation_numerical = dict()
    for col in column_numerical:
        tmp = f'{df[col].min()} - {df[col].max()}'
        variation_numerical[col] = tmp
    tmp = pd.Series(variation_numerical)
    data_variation_numerical = pd.DataFrame(tmp).T.rename({0: 'data variation'})
    data_variation = pd.concat([data_variation_numerical.rename({'data variation': 'range'}),
                                data_variation_categorical.rename({'data variation': 'variation'})], axis=1).fillna('-')
    data_viz = df.copy()
    data_viz.loc[:, 'Attrition'] = data_viz.loc[:, 'Attrition'].apply(lambda x: 'Attrition' if x == 1 else 'retain')

    columns = data_viz['Attrition']

    attr_crosstab = pd.DataFrame()

    for col in column_categorical:  # column_categorical
        # create crosstab for each attribute
        index = data_viz[col]
        ct = pd.crosstab(index=index, columns=columns, normalize='index', colnames=[None]).reset_index()

        # add prefix to each category
        # format: column name (category)
        col_titleize = inflection.titleize(col)
        ct[col] = ct[col].apply(lambda x: f'{col_titleize} ({x})')

        # rename the column
        ct.rename(columns={col: 'attribute'}, inplace=True)

        # create a single dataframe
        attr_crosstab = pd.concat([attr_crosstab, ct])

    attr_crosstab = attr_crosstab.sort_values('Attrition', ascending=False).reset_index(drop=True)
    data_X = df.drop('Attrition', axis=1)
    data_y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3,
                                                        random_state=1, stratify=data_y)

    ######################
    # Label Encoding
    # I use it for the target variable (label).
    ######################

    le = LabelEncoder()

    le.fit(y_train)
    y_train_encode = le.transform(y_train)
    y_test_encode = le.transform(y_test)

    # drop 1 category if the feature only has 2 categories
    ohe = OneHotEncoder(sparse=False, drop='if_binary')
    ohe.fit(X_train[column_categorical])
    X_train_ohe = ohe.transform(X_train[column_categorical])
    X_test_ohe = ohe.transform(X_test[column_categorical])
    column_ohe = ohe.get_feature_names_out()

    # create dataframe from one-hot encoded features
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=column_ohe, index=X_train.index)
    # combine the numerical and encoded features
    X_train_encode = pd.concat([X_train.drop(columns=column_categorical), X_train_ohe_df], axis=1)
    # create dataframe from one-hot encoded features
    X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=column_ohe, index=X_test.index)
    # combine the numerical and encoded features
    X_test_encode = pd.concat([X_test.drop(columns=column_categorical), X_test_ohe_df], axis=1)
    # combine the X-train and X-test
    data_encode = pd.concat([X_train_encode, X_test_encode], axis=0)
    # combine with the y-train
    data_encode = data_encode.join(pd.Series(y_train_encode, name='Attrition', index=X_train_encode.index),
                                   lsuffix='_1', rsuffix='_2')
    # combine with the y-test
    data_encode = data_encode.join(pd.Series(y_test_encode, name='Attrition', index=X_test_encode.index), lsuffix='_1',
                                   rsuffix='_2')
    # merging the y-train and y-test column
    data_encode['Attrition_1'].fillna(data_encode['Attrition_2'], inplace=True)
    data_encode.drop(columns='Attrition_2', inplace=True)
    data_encode.rename(columns={'Attrition_1': 'Attrition'}, inplace=True)

    # numerical
    df1 = df
    df1.drop('Attrition', axis=1, inplace=True)
    column_numerical1 = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_train_scale = X_train_encode.copy()
    X_test_scale = X_test_encode.copy()

    for i in column_numerical1:
        scaler = MinMaxScaler()
        scaler.fit(X_train_scale[[i]])

        X_train_scale[[i]] = scaler.transform(X_train_scale[[i]])
        X_test_scale[[i]] = scaler.transform(X_test_scale[[i]])

    # combine the X-train and X-test
    data_scale = pd.concat([X_train_scale, X_test_scale], axis=0)
    # combine with the y-train
    data_scale = data_scale.join(pd.Series(y_train_encode, name='Attrition', index=X_train_scale.index), lsuffix='_1',
                                 rsuffix='_2')
    # combine with the y-test
    data_scale = data_scale.join(pd.Series(y_test_encode, name='Attrition', index=X_test_scale.index), lsuffix='_1',
                                 rsuffix='_2')
    # merging the y-train and y-test column
    data_scale['Attrition_1'].fillna(data_scale['Attrition_2'], inplace=True)
    data_scale.drop(columns='Attrition_2', inplace=True)
    data_scale.rename(columns={'Attrition_1': 'Attrition'}, inplace=True)

    data_scale_train = pd.concat(
        [X_train_scale, pd.Series(y_train_encode, name='Attrition', index=X_train_scale.index)], axis=1)

    corr_matrix = data_scale.corr().round(3)
    corr_target = corr_matrix['Attrition'].drop('Attrition')

    smote = SMOTE(random_state=1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scale, y_train_encode)
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_train_smote.columns)
    y_train_smote_df = pd.DataFrame(y_train_smote, columns=['Attrition'])
    data_smote = pd.concat([X_train_smote_df, y_train_smote_df], axis=1)
    numeric_cols1 = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    column_numerical2 = ['Age',
                         'DailyRate',
                         'DistanceFromHome', ]
    X_train_model = X_train_smote.copy()
    y_train_model = y_train_smote.copy()

    X_test_model = X_test_scale.copy()
    y_test_model = y_test_encode.copy()
    return X_train_model, y_train_model, X_test_model, y_test_model,X_train_scale,X_test_scale


def get_score(y_pred_list, y_test, average=None, plot=True, axis=0, cmap='Blues'):
    model_name = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []

    for name, y_pred in y_pred_list.items():
        model_name.append(name)
        if average != None:
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
    # if plot:
    #    print(score_df.style.background_gradient(axis=axis, cmap=cmap))
    return score_df


def model_compare():
    X_train_model, y_train_model, X_test_model, y_test_model,X_train_scale,X_test_scale = Model_create1()
    model_list = {
        'Random Forest': RandomForestClassifier(random_state=1),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=1),
        'AdaBoost Classifier': AdaBoostClassifier(random_state=1),
        'CatBoost Classifier': CatBoostClassifier(random_state=1, verbose=False),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=1),
        'XGBoost': XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=1),
    }
    # Reads in saved model
    y_pred_list = pickle.load(open('saved_model1.pkl', 'rb'))
    score_smote = get_score(y_pred_list, y_test_model, average='macro')
    threshold = 13
    filter = SelectKBest(score_func=f_classif, k=threshold)
    filter.fit(X_train_model, y_train_model)
    X_train_filter = filter.transform(X_train_model)
    X_test_filter = filter.transform(X_test_model)

    # estimator parameter:
    # A supervised learning estimator with a fit method
    # that provides information about feature importance
    # (e.g. coef_, feature_importances_) -> DecisionTree uses feature importance

    threshold = 13
    estimator = LogisticRegression(random_state=1)
    wrapper = RFE(estimator=estimator, n_features_to_select=threshold)
    wrapper.fit(X_train_model, y_train_model)
    X_train_wrap = wrapper.transform(X_train_model)
    X_test_wrap = wrapper.transform(X_test_scale)
    # st.write('before wrapper\t:', X_train_model.shape)
    # st.write('after wrapper\t:', X_train_wrap.shape)
    y_pred_list = dict()

    for name, model in model_list.items():
        model.fit(X_train_wrap, y_train_model)
        y_pred_list[name] = model.predict(X_test_wrap)

    score_wrap = get_score(y_pred_list, y_test_model, average='macro')

    # estimator parameter:
    # A supervised learning estimator with a fit method
    # that provides information about feature importance
    # (e.g. coef_, feature_importances_) -> DecisionTree uses feature importance

    estimator = LogisticRegression(random_state=1)
    embedded = SelectFromModel(estimator=estimator, threshold='median')
    embedded.fit(X_train_model, y_train_model)
    X_train_embed = embedded.transform(X_train_model)
    X_test_embed = embedded.transform(X_test_scale)
    # st.write('before embedded\t:', X_train_model.shape)
    # st.write('after embedded\t:', X_train_embed.shape)

    y_pred_list = dict()
    for name, model in model_list.items():
        model.fit(X_train_embed, y_train_model)
        y_pred_list[name] = model.predict(X_test_embed)
    score_embed = get_score(y_pred_list, y_test_model, average='macro')

    y_pred_list = dict()

    for name, model in model_list.items():
        model.fit(X_train_filter, y_train_model)
        y_pred_list[name] = model.predict(X_test_filter)

    score_filter = get_score(y_pred_list, y_test_model, average='macro')

    score_smote_mean = pd.DataFrame(score_smote.mean(), columns=['original']).T
    score_filter_mean = pd.DataFrame(score_filter.mean(), columns=['filter method']).T
    score_wrap_mean = pd.DataFrame(score_wrap.mean(), columns=['wrapper method']).T
    score_embed_mean = pd.DataFrame(score_embed.mean(), columns=['embedded method']).T

    score_compare = pd.concat([score_smote_mean,
                               score_filter_mean,
                               score_wrap_mean,
                               score_embed_mean], axis=0)
    # save dataframe to pickle file
    score_compare.to_pickle('score_compare.pkl')


#model_compare()

# read pickle file as dataframe
# df_sales = pd.read_pickle('score_compare.pkl')
# # display the dataframe
# print(df_sales)
