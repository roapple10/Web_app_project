import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
# viz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

plt.style.use('fivethirtyeight')
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 1.5})
# change the maximum width in characters of a column (default: 50)
pd.set_option('display.max_colwidth', None)
# change the display precision for better viz
pd.set_option('display.precision', 3)


######################
# Pie plot
######################

def pie(pie_data, pie_label):
    # pie_data = data_viz['Attrition'].value_counts(normalize=True).values * 100
    # pie_label = data_viz['Attrition'].value_counts(normalize=True).index.to_list()

    fig, ax = plt.subplots(figsize=(2, 6))

    wedges, texts, autotexts = ax.pie(pie_data, labels=pie_label,
                                      startangle=90, explode=[0, 0.1],
                                      autopct='%.0f%%',
                                      textprops={'color': 'w', 'fontsize': 8, 'weight': 'bold'})

    for i, wedge in enumerate(wedges):
        texts[i].set_color(wedge.get_facecolor())

    plt.tight_layout()
    st.pyplot()


def bar_plot(attr_crosstab):
    attr_crosstab.style.background_gradient()
    k = 'Attrition'
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=attr_crosstab.iloc[:5], x=k, y='attribute', ax=ax, palette=['#FC4F30'], saturation=1)
    # ax.bar_label(ax.containers[0], padding=3, fmt='%.2f', fontsize=14, fontweight='medium')
    ax.grid(False, axis='y')
    ax.set_title('Top 5 Categories with the Highest Probability to Attrition')
    ax.set_xlim(0, 1)
    ax.set_ylabel('')
    ax.set_xlabel('Attrition Probability')
    ax.set_xticklabels([])
    sns.despine(left=True, bottom=True)
    st.pyplot()
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=attr_crosstab.iloc[-5:].sort_values(k), x=k, y='attribute', ax=ax, palette=['#008FD5'],
                saturation=1)
    # ax.bar_label(ax.containers[0], padding=3, fmt='%.2f', fontsize=14, fontweight='medium')
    ax.grid(False, axis='y')
    ax.set_title('Top 5 Categories with the Lowest Probability to Attrition')
    ax.set_xlim(0, 1)
    ax.set_ylabel('')
    ax.set_xlabel('Attrition Probability')
    ax.set_xticklabels([])
    sns.despine(left=True, bottom=True)
    # plt.show()
    st.pyplot()


def bar_plot1(corr_matrix, corr_target):
    fig, ax = plt.subplots(figsize=(12, 18))
    sns.barplot(x=corr_target.values, y=corr_target.index, ax=ax)
    ax.bar_label(ax.containers[0], padding=3, fmt='%.2f', fontsize=14, fontweight='medium')
    ax.axis('tight')
    sns.despine(left=True)
    st.pyplot()


def bar_plot2(selected_feature, feature_name, feature_score, X_train_model, threshold):
    # add to dataframe
    feature_selection = pd.DataFrame({'feature_name': feature_name, 'feature_score': feature_score}).sort_values(
        'feature_score', ascending=False)
    feature_selection.reset_index(drop=True, inplace=True)

    # create flag
    selected = list()
    for i in range(0, X_train_model.shape[1]):
        if i < threshold:
            selected.append(True)
        else:
            selected.append(False)

    selected_s = pd.Series(selected, name='selected')
    feature_selection = pd.concat([feature_selection, selected_s], axis=1)
    # highlight the top features
    palette = []
    for i in range(0, len(feature_selection)):
        if i < threshold:
            palette.append('#008FD5')
        else:
            palette.append('silver')
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.barplot(data=feature_selection, x='feature_score', y='feature_name', ax=ax, palette=palette)
    # ax.bar_label(ax.containers[0], padding=3, fmt='%.2f', fontsize=14, fontweight='medium')
    # custom y label color
    for i, label in enumerate(ax.yaxis.get_ticklabels()):
        if feature_selection.loc[i, 'selected'] == False:
            label.set_color('silver')
    # custom bar label visibility
    for con in ax.containers:
        # labels = [val for val in con.datavalues]
        labels = con.datavalues
        labels_len = len(labels)

        # masking the top features
        np.put(labels, np.arange(threshold, labels_len), [-1])

        # hide the labels for non-top features
        labels = [f'{val:.2f}' if val != -1 else '' for val in labels]

        ax.bar_label(con, labels=labels, padding=3, fontsize=14, fontweight='medium')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.despine(left=True, bottom=True)
    st.pyplot()


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

    if plot:
        st.write(score_df.style.background_gradient(axis=axis, cmap=cmap))

    return score_df
