# ==============================
# Import Essential Libraries
# ==============================

# General-purpose libraries
import re
import time
import random
import string
import emoji
from collections import Counter
from itertools import combinations
from copy import deepcopy

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from prettytable import PrettyTable, HRuleStyle, VRuleStyle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns

# Text processing and NLP
import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Statistical and correlation analysis
from scipy.stats import pearsonr

# Machine Learning Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, classification_report, mean_squared_error, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

# Import TextBlob for sentiment analysis
from textblob import TextBlob

# Network graph libraries
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities 
from adjustText import adjust_text
import matplotlib.cm as cm 

# ==========================================================================================================================================
# Data Manipulation
# ==========================================================================================================================================

def top_words(data, column_name = None, top_n = 20):
    """
    Extracts the top N most frequent words from a given dataset.

    Parameters:
        data (pd.DataFrame or list): The input data. If 'column_name' is specified, 'data' should be a pandas DataFrame. 
                                     If 'column_name' is None, 'data' should be a list or a single string containing words.
        column_name (str, optional): The name of the column in the DataFrame to extract text data from. Defaults to None.
        top_n (int, optional): The number of top frequent words to retrieve. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing two columns: 
                      - 'Word': The unique words.
                      - 'Frequency': The count of each word, sorted in descending order.
    """
    if column_name:
        all_words = ' '.join(data[column_name]).split()
    else:
        all_words = data
        
    word_freq = pd.Series(all_words).value_counts().head(top_n)
    word_freq_df = word_freq.reset_index()
    word_freq_df.columns = ['Word', 'Frequency']

    return word_freq_df

def generate_word_frequencies_for_cuisine(data, cuisine_type):
    """
    This function generates word frequencies for a specific cuisine type from the dataset.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame where each row represents a data entry and contains a 'Cuisine Type' 
                             column along with other columns representing word counts or features.
        cuisine_type (str): The specific cuisine type to filter the data by.

    Returns:
        dict: A dictionary where the keys are column names (representing words or features) and the values are their 
             corresponding summed frequencies for the specified cuisine type.
    """
    cuisine_df = data[data['Cuisine Type'] == cuisine_type]
    word_frequencies = cuisine_df.drop('Cuisine Type', axis = 1).sum().to_dict() 
    return word_frequencies

def find_matching_rows(row):
    """
    This function checks if any of the cuisines listed in the 'Cuisines' column are mentioned in the 'Review' column.

    Parameters:
        row (pd.Series): A single row of the pandas DataFrame, typically passed during a DataFrame's `apply` function. 
                         This row should have two columns: 'Review' (a text string) and 'Cuisines' (a comma-separated list of cuisines).

    Returns:
        bool: True if any of the cuisines in the 'Cuisines' column are mentioned in the 'Review' column, otherwise False.
    """
    return any(cuisine in row['Review'] for cuisine in row['Cuisines'].split(', '))

def treat_matching_rows(row):
    """
    Removes any cuisines listed in the 'Cuisines' column from the 'Lemmas_Treated' column of a row in a pandas DataFrame.

    Parameters:
        row (pd.Series): A single row of the pandas DataFrame, typically passed during a DataFrame's `apply` function. 
                         This row should have:
                         - 'Lemmas_Treated': A string containing the processed review text.
                         - 'Cuisines': A comma-separated list of cuisines.

    Returns:
        str: The modified 'Lemmas_Treated' string with any occurrences of cuisines listed in the 'Cuisines' column removed.
    """
    review = row['Lemmas_Treated']
    for cuisine in row['Cuisines'].split(', '):
        review = re.sub(r'\b' + re.escape(cuisine) + r'\b', '', review)
    return review.strip()

def clean_nan_values(X, y):
    """
    Cleans NaN values from two corresponding datasets, X and y, by removing entries in X that contain NaN values, 
    and their corresponding values in y.

    Parameters:
        X (iterable): An iterable (e.g., list, array) where each element is a feature vector or set of values. 
                     NaN values within these feature vectors will be identified.
        y (iterable): An iterable (e.g., list, array) containing the labels or target values corresponding to the elements in X.

    Returns:
        tuple: A tuple containing:
               - X_clean (np.ndarray): The cleaned version of X with rows containing NaN values removed.
               - y_clean (np.ndarray): The corresponding cleaned version of y.
    """
    combined_df = pd.DataFrame({'X_values': list(X), 'y_values': list(y)})

    nan_mask = combined_df['X_values'].apply(lambda x: np.isnan(x).any())

    cleaned_df = combined_df[~nan_mask].reset_index(drop = True)

    X_clean = np.array(cleaned_df['X_values'].tolist())
    y_clean = np.array(cleaned_df['y_values'].tolist())

    return X_clean, y_clean

def calculate_class_weights(y_train):
    """
    Calculates class weights to address class imbalance in a training dataset.

    Parameters:
        y_train (array-like): A 1D or flattened array containing the class labels for the training data.

    Returns:
        dict: A dictionary where the keys are the unique class labels and the values are their corresponding computed weights. 
             These weights are calculated to balance the classes based on their frequencies in the training dataset.
    """
    unique_classes = np.unique(y_train.flatten())
    
    class_weights = compute_class_weight('balanced', classes = unique_classes, y = y_train.flatten())
    
    class_weight_dictionary = {unique_classes[i]: class_weights[i] for i in range(len(unique_classes))}
    
    return class_weight_dictionary

def compute_pearson_correlation(dataframe, column_1, column_2):
    """
    Computes the Pearson correlation coefficient between two columns in a pandas DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        column_1 (str): The name of the first column to use in the correlation calculation.
        column_2 (str): The name of the second column to use in the correlation calculation.

    Returns:
        float: The Pearson correlation coefficient between the two specified columns, which ranges from -1 
             (perfect negative correlation) to 1 (perfect positive correlation). A value of 0 indicates no linear correlation.
    """  
    correlation = dataframe[column_1].corr(dataframe[column_2], method = 'pearson')
    return correlation

# ==========================================================================================================================================
# Data Visualization
# ==========================================================================================================================================

def set_plot_properties(fig, x_label, y_label, y_lim = []):
    """
    Sets properties for a plotly figure, including axis labels and optional y-axis limits.

    Parameters:
        fig (plotly.graph_objects.Figure): The plotly figure to update.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): A list of two values specifying the lower and upper limits for the y-axis. 
                                Defaults to an empty list, which means no limit is applied.

    Returns:
        None: The function modifies the input figure in place.
    """
    fig.update_layout(xaxis_title = x_label, yaxis_title = y_label)
    if len(y_lim) == 2:
        fig.update_yaxes(range = y_lim)

def bar_plot(data, variable, x_label, y_label = 'Count', y_lim = [], legend = [], color = 'rgb(141, 160, 203)', annotate = False, top = None):
    """
    Creates and displays a bar plot for a specified variable from a dataset.

    Parameters:
        data (pd.DataFrame): The input dataset containing the variable to plot.
        variable (str): The name of the column in the dataset to visualize.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): A list of two values specifying the lower and upper limits for the y-axis. 
                                Defaults to an empty list (no limits applied).
        legend (list, optional): A list for the legend (currently unused but can be extended). Defaults to an empty list.
        color (str, optional): The color for the bars. Defaults to 'rgb(141, 160, 203)'.
        annotate (bool, optional): If True, adds annotations with the counts on top of the bars. Defaults to False.
        top (int, optional): If provided, limits the plot to the top N most frequent categories. 
                             Defaults to None (all categories are included).

    Returns:
        None: The function directly displays the bar plot using Plotly.
    """
    counts = data[variable].value_counts()[:top] if top else data[variable].value_counts()
    x = counts.index  
    y = counts.values  

    fig = px.bar(x = x, y = y, labels = {'x': x_label, 'y': y_label}, color_discrete_sequence = [color])

    if annotate:
        for i, value in enumerate(y):
            fig.add_annotation(x = x[i], y = value, text = str(value), showarrow = False, font = dict(size = 12))
    
    set_plot_properties(fig, x_label, y_label, y_lim)

    fig.show()

def donut_chart(words_df, title):
    """
    Creates and displays a donut chart to visualize word frequencies.

    Parameters:
        words_df (pd.DataFrame): A DataFrame containing two columns:
                                 - 'Word': The labels for the chart (e.g., words or categories).
                                 - 'Frequency': The values corresponding to each label.
        title (str): The title of the chart.

    Returns:
        None: The function directly displays the donut chart using Plotly.
    """
    fig = go.Figure(data = [go.Pie(labels = words_df['Word'], values = words_df['Frequency'], hole = 0.4, textinfo = 'label + percent', marker = dict(colors = px.colors.sequential.Blues))])

    fig.update_layout(title_text = title, title_font_size = 24, title_x = 0.5,  showlegend = False,  width = 1000,  height = 800)
    
    fig.show()

def word_cloud(frequencies):
    """
    Generates a word cloud from a dictionary of word frequencies.

    Parameters:
        frequencies (dict): A dictionary where the keys are words (str) and the values are their corresponding 
                            frequencies (int or float).

    Returns:
        WordCloud: A WordCloud object containing the generated word cloud, which can be displayed using Matplotlib 
                 or other visualization tools.
    """
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white', colormap = 'Blues').generate_from_frequencies(frequencies)
    return wordcloud 

def plot_histogram(dataframe, x, nbins = 20, title = 'Histogram', labels = None, xaxis_title = 'X-Axis', yaxis_title = 'Y-Axis'):
    """
    Creates and displays a histogram for a specified column in a pandas DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the data to plot.
        x (str): The name of the column to be used for the histogram.
        nbins (int, optional): The number of bins for the histogram. Defaults to 20.
        title (str, optional): The title of the histogram. Defaults to 'Histogram'.
        labels (dict, optional): A dictionary for renaming axes labels, where keys are column names and values are their corresponding 
                                 labels. Defaults to None.
        xaxis_title (str, optional): The label for the x-axis. Defaults to 'X-Axis'.
        yaxis_title (str, optional): The label for the y-axis. Defaults to 'Y-Axis'.

    Returns:
        None: The function directly displays the histogram using Plotly.
    """
    fig = px.histogram(dataframe, x = x, nbins = nbins, title = title, labels = labels, color_discrete_sequence = ['rgb(141, 160, 203)'])

    fig.update_layout(xaxis_title = xaxis_title, yaxis_title = yaxis_title, template = 'plotly_white')

    fig.show()

def line_plot(data, x_col, y_col, title):
    """
    Creates and displays a line plot for specified x and y columns in a dataset.

    Parameters:
        data (pd.DataFrame): The input dataset containing the data to plot.
        x_col (str): The column name to use for the x-axis.
        y_col (str): The column name to use for the y-axis.
        title (str): The title of the line plot.

    Returns:
        None: The function directly displays the line plot using Plotly.
    """
    fig = px.line(data, x = x_col, y = y_col, title = title)
    fig.update_traces(line = dict(color = 'rgb(141, 160, 203)'), fill = 'tozeroy')
    fig.update_xaxes(tickmode = 'linear', tick0 = data[x_col].min(), dtick = 1)
    fig.show()
    
def plot_treemap(lemmas_column, pos_tags_column, sample_size = 20, title = 'Treemap of Words by POS Tag'):
    """
    Creates and displays a treemap of words categorized by their part-of-speech (POS) tags.

    Parameters:
        lemmas_column (pd.Series): A pandas Series containing lists of lemmas (words) for each document or text entry.
        pos_tags_column (pd.Series): A pandas Series containing lists of part-of-speech (POS) tags corresponding to the lemmas.
        sample_size (int, optional): The number of samples to select from the lemmas and POS tags columns for the treemap. Defaults to 20.
        title (str, optional): The title of the treemap. Defaults to 'Treemap of Words by POS Tag'.

    Returns:
        None: The function directly displays the treemap using Plotly.
    """
    sampled_lemmas = lemmas_column.sample(n = sample_size, random_state = 42).reset_index(drop = True)
    sampled_pos_tags = pos_tags_column.sample(n = sample_size, random_state = 42).reset_index(drop = True)

    list_of_words = []

    for lemmas, pos_tags in zip(sampled_lemmas, sampled_pos_tags):
        for lemma, pos in zip(lemmas, pos_tags):
            list_of_words.append({"Word": lemma, "POS Tag": pos})

    words_to_treemap = pd.DataFrame(list_of_words)

    fig = px.treemap(words_to_treemap, path = ['POS Tag', 'Word'], color = 'POS Tag', title = title)
    fig.show()

def plot_word_count_comparison(data, lemmas, lemmas_treated):
    """
    Compares the total word count between two columns: 'lemmas' and 'lemmas_treated' in a dataset.

    Parameters:
        data (pd.DataFrame): The input dataset containing the columns for 'lemmas' and 'lemmas_treated'.
        lemmas (str): The name of the column containing the original text or lemmas.
        lemmas_treated (str): The name of the column containing the treated or processed text.

    Returns:
        None: The function directly displays a bar chart comparing the total word count in 'lemmas' and 'lemmas_treated' using Plotly.
    """
    lemmas_total_word_count = data[lemmas].apply(lambda x: len(x.split())).sum()
    lemmas_treated_total_word_count = data[lemmas_treated].apply(lambda x: len(x.split())).sum()

    word_counts = {'Category': ['Lemmas', 'Lemmas_Treated'],'Word Count': [lemmas_total_word_count, lemmas_treated_total_word_count]}

    fig = go.Figure(data=[go.Bar(x = word_counts['Category'], y = word_counts['Word Count'], marker = dict(color = ['rgb(141, 160, 203)', 'rgb(179, 205, 227)']))])

    fig.update_layout(title = 'Total Word Count: Lemmas vs Lemmas_Treated', xaxis_title = 'Category', yaxis_title = 'Total Word Count', template = 'plotly', showlegend = False)

    fig.show()

def PCA_Scatter_Plot_1(model, num_words = 50):
    """
    Generates and displays a 2D scatter plot of word embeddings using PCA (Principal Component Analysis) for dimensionality reduction.

    Parameters:
        model (gensim.models.KeyedVectors): A trained word embedding model (e.g., Word2Vec or GloVe) from which embeddings are extracted.
        num_words (int, optional): The number of top words to visualize in the PCA scatter plot. Defaults to 50.

    Returns:
        None: The function directly displays a scatter plot using Plotly.
    """
    embedding_df = embedding_function(model, model.wv.index_to_key[:num_words])
    
    fig = px.scatter(embedding_df, x = 'PCA_dim_1', y = 'PCA_dim_2', title = f'PCA of Word Embeddings (Top {num_words} Words)', text = 'Words')
    
    fig.update_traces(textposition = 'top center', marker = dict(size = 10, color = 'black'), textfont = dict(color = 'black'))
    fig.update_layout(xaxis_title = 'PCA Dimension 1', yaxis_title = 'PCA Dimension 2', plot_bgcolor = 'rgb(141, 160, 203)', showlegend = False,  xaxis = dict(showgrid = True, gridcolor = 'white'), yaxis = dict(showgrid = True, gridcolor = 'white'), coloraxis_showscale = False)
    
    fig.show()

def plot_model_performance(one_vs_rest_metrics, classifier_chain_metrics, vectorizer_name, models = ["LogisticRegression", "DecisionTree", "MLPClassifier", "RandomForest", "Dummy"], metrics = ["F1 Score", "Precision", "Recall"]):
    """
    Plots the performance comparison of different models using the OneVsRest and ClassifierChain strategies.

    Parameters:
        one_vs_rest_metrics (dict): A dictionary where keys are model names and values are dictionaries of performance metrics 
                                    (e.g., F1 Score, Precision, Recall) for each model using the OneVsRest strategy.
        classifier_chain_metrics (dict): A dictionary where keys are model names and values are dictionaries of performance metrics 
                                         for each model using the ClassifierChain strategy.
        vectorizer_name (str): The name of the vectorizer used in the model training (e.g., 'TF-IDF', 'CountVectorizer').
        models (list, optional): A list of model names to include in the plot. Defaults to ["LogisticRegression", "DecisionTree", "MLPClassifier", "RandomForest", "Dummy"].
        metrics (list, optional): A list of metrics to display in the plot (e.g., ["F1 Score", "Precision", "Recall"]). Defaults to ["F1 Score", "Precision", "Recall"].

    Returns:
        None: The function directly displays the grouped bar chart comparing model performance for the two strategies using Plotly.
    """
    colors = ["#8DA0CB", "#B3CDE3", "#316395"]
    
    fig = make_subplots(rows = 1, cols = 2, shared_yaxes = True, subplot_titles = ("OneVsRest Classifier Performance", "ClassifierChain Performance"))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [one_vs_rest_metrics[model][metric] for model in models]
        fig.add_trace(go.Bar(name = metric, x = models, y = values, marker_color = color, offsetgroup = i),row = 1, col = 1)
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [classifier_chain_metrics[model][metric] for model in models]
        fig.add_trace(go.Bar(name = metric, x = models, y = values, marker_color = color, offsetgroup = i, showlegend = False), row = 1, col = 2)
    
    fig.update_layout(
        title = dict(text = f"<b>Model Performance Using {vectorizer_name} Vectorizer</b>",  x = 0.5, font = dict(size = 16)),
        barmode = "group", legend_title = "Metrics", xaxis = dict(tickangle = 0),   xaxis2 = dict(tickangle = 0),  yaxis_title = "Score", template = "plotly_white", width = 1500, height = 600)
    
    fig.show()

def check_overfitting(model, X_train, y_train, X_val, y_val):
    """
    Checks for overfitting by comparing the performance of a model on training and validation data using different metrics.

    Parameters:
        model (sklearn.base.BaseEstimator): The trained model to evaluate.
        X_train (array-like or pd.DataFrame): The feature data used for training the model.
        y_train (array-like or pd.Series): The target labels for training data.
        X_val (array-like or pd.DataFrame): The feature data used for validation.
        y_val (array-like or pd.Series): The target labels for validation data.

    Returns:
        None: The function directly displays bar plots comparing the model's performance on training and validation sets for three metrics: F1 score, precision, and recall.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metric_funcs = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    metrics = ["f1", "precision", "recall"]
    
    average = "micro" if len(y_train.shape) > 1 else None

    fig, axes = plt.subplots(1, 3, figsize = (18, 6))

    train_color = (141/255, 160/255, 203/255)  
    val_color = (179/255, 205/255, 227/255)  

    for i, metric in enumerate(metrics):
        metric_func = metric_funcs[metric]

        train_score = metric_func(y_train, y_train_pred, average = average)
        val_score = metric_func(y_val, y_val_pred, average = average)

        ax = axes[i]
        ax.bar("Training", train_score, color = train_color)
        ax.bar("Validation", val_score, color = val_color)

        ax.text("Training", train_score - 0.05, f"{train_score:.4f}", ha = "center", va = "center", color = "black", fontsize = 12, fontweight = 'bold')
        ax.text("Validation", val_score - 0.05, f"{val_score:.4f}", ha = "center", va = "center", color = "black", fontsize = 12, fontweight = 'bold')

        ax.set_title(f"{metric.capitalize()} Score", fontsize = 14, fontweight = 'bold')
        ax.set_ylim([0, 1])  
        ax.set_ylabel(f"{metric.capitalize()} Score", fontsize = 12)

    plt.tight_layout()
    plt.show()

def F1_Scores_Cuisine_Types(f1_scores_df):
    """
    Visualizes the F1 scores of different cuisine types for both training and validation sets using a grouped bar chart.

    Parameters:
        f1_scores_df (pd.DataFrame): A DataFrame where each row corresponds to a cuisine type, and contains the F1 scores for training 
                                     and validation sets. The DataFrame should have the following columns: 'F1 Score (Train)', 
                                     'F1 Score (Validation)'.

    Returns:
        None: The function directly displays a grouped bar chart comparing F1 scores for training and validation sets per cuisine 
             type using Plotly.
    """
    f1_scores_train = f1_scores_df['F1 Score (Train)']
    f1_scores_validation = f1_scores_df['F1 Score (Validation)']

    fig = go.Figure()

    fig.add_trace(go.Bar(x = f1_scores_df.index  , y = f1_scores_train, name = 'F1 Score (Train)', marker_color = '#8DA0CB', text = f1_scores_train.round(2), textposition = 'outside'))

    fig.add_trace(go.Bar(x = f1_scores_df.index  , y = f1_scores_validation, name = 'F1 Score (Validation)', marker_color = '#B3CDE3', text = f1_scores_validation.round(2), textposition = 'outside'))

    fig.update_layout(title = dict(text = "F1 Scores per Cuisine Type (Train vs. Validation)", x = 0.5, font = dict(size = 18)),
        xaxis = dict(title = "Cuisine Type", tickangle = -45, tickfont = dict(size = 10), rangeslider = dict(visible = False)),
        yaxis = dict(title = "F1 Score", tickfont = dict(size = 10), range = [0, 1]),
        barmode = 'group', template = "plotly_white", width = 1800, height = 600, margin = dict(t = 50, b = 150))

    fig.update_xaxes(automargin = True, showline = True, fixedrange = False)

    fig.show()

def plot_correlation_comparison(dataframe, compound_score_column = 'Compound Score (VADER)', mean_compound_score_column = 'Mean Compound Score (VADER)'):
    """
    Plots a scatter plot to compare the relationship between the 'Compound Score (VADER)' and the 'Mean Compound Score (VADER)', 
    with a reference line indicating perfect correlation.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the columns for the compound scores and the mean compound scores.
        compound_score_column (str, optional): The name of the column representing the compound score 
                                             (default is 'Compound Score (VADER)').
        mean_compound_score_column (str, optional): The name of the column representing the mean compound score 
                                                    (default is 'Mean Compound Score (VADER)').

    Returns:
        None: The function directly displays a scatter plot comparing the two compound scores, with a reference line for 
             perfect correlation.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = dataframe[compound_score_column],
        y = dataframe[mean_compound_score_column],
        mode = 'markers',
        marker = dict(color = 'rgb(141, 160, 203)', opacity = 0.6), showlegend = False))

    fig.add_trace(go.Scatter(
        x = [dataframe[compound_score_column].min(), dataframe[compound_score_column].max()],
        y = [dataframe[compound_score_column].min(), dataframe[compound_score_column].max()],
        mode = 'lines',
        line = dict(color = 'black', dash = 'dash'),
        name = 'Perfect Correlation'))

    fig.update_layout(
        title = 'Comparison: Compound Score vs. Mean Compound Score',
        xaxis_title = 'Compound Score',
        yaxis_title = 'Mean Compound Score',
        showlegend = True,
        template = 'plotly_white',
        font = dict(size = 14),
        width = 900, height = 600,
        legend=dict(x = 1, y = 1, traceorder = 'normal', orientation = 'h', xanchor = 'right', yanchor  = 'bottom'))

    fig.show()

def plot_histogram_VADER(dataframe, positive_column, negative_column, neutral_column, compound_column, num_bins = 20):
    """
    Plots histograms to visualize the distribution of positive, neutral, and negative compound scores for a VADER sentiment analysis output.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the VADER sentiment scores, including positive, negative, neutral, 
                                 and compound columns.
        positive_column (str): The name of the column containing the positive sentiment scores.
        negative_column (str): The name of the column containing the negative sentiment scores.
        neutral_column (str): The name of the column containing the neutral sentiment scores.
        compound_column (str): The name of the column containing the compound sentiment scores (though not directly used in the histogram).
        num_bins (int, optional): The number of bins for the histograms (default is 20).

    Returns:
        None: The function directly displays three histograms comparing the distribution of positive, neutral, and negative sentiment scores.
    """
    fig = make_subplots(rows = 1, cols = 3,
        subplot_titles = [
            "Positive Compound", "Neutral Compound", "Negative Compound"],
        horizontal_spacing = 0.15, vertical_spacing = 0.2)
    
    bin_edges = np.linspace(0, 1, num_bins + 1)  

    for idx, (col, row, col_num, score_type) in enumerate([
        (positive_column, 1, 1, compound_column), (neutral_column, 1, 2, compound_column), (negative_column, 1, 3, compound_column)]):

        hist_values, _ = np.histogram(dataframe[col], bins = bin_edges)

        fig.add_trace(go.Bar(x = bin_edges[:-1], y = hist_values, marker_color = 'rgb(141, 160, 203)', name = f"{col} Histogram"), row = row, col = col_num)

    fig.update_layout(
        title = "Distribution of Positive, Neutral and Negative Scores",
        title_font = dict(family = "Arial", size = 16, weight = "bold"),
        showlegend = False, template = "plotly_white",
        font = dict(size = 14), height = 600, width = 2000)

    fig.show()

def plot_heatmap_VADER(dataframe, columns = ['Compound Score (VADER)', 'Positive', 'Neutral', 'Negative']):
    """
    Plots a heatmap to visualize the correlation between different sentiment scores (such as compound, positive, neutral, and negative) 
    from a VADER sentiment analysis.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing sentiment scores for various sentiment categories, such as compound, 
                                 positive, neutral, and negative.
        columns (list of str, optional): The columns to be included in the correlation heatmap. By default, it includes the compound, 
                                         positive, neutral, and negative sentiment scores.

    Returns:
        None: The function directly displays a heatmap showing the correlation between the specified sentiment scores.
    """
    correlation_matrix = dataframe[columns].corr()

    mask = np.tril(np.ones_like(correlation_matrix, dtype = bool), k = -1) 
    correlation_matrix[mask] = np.nan

    fig = go.Figure(data = go.Heatmap(z = correlation_matrix.values, x = correlation_matrix.columns, y = correlation_matrix.columns,
        colorscale = 'Blues', colorbar = dict(title = "Correlation"), text = np.where(np.isnan(correlation_matrix.values), '', correlation_matrix.round(2)), texttemplate = "%{text}"))

    fig.update_layout(title = "Correlation Heatmap Between Sentiment Scores", xaxis_title = "Sentiment Features", yaxis_title = "Sentiment Features", template = "plotly_white", font = dict(size = 14))

    fig.show()

def score_per_cuisine(data, vader = True):
    """
    Plots histograms for sentiment scores (VADER or TextBlob) for each cuisine type in the dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing sentiment scores and cuisine types.
        vader (bool, optional): If True, uses VADER's compound score ('Compound Score (VADER)'). 
                                If False, uses TextBlob's polarity score ('Polarity Score (TextBlob)'). Default is True.

    Returns:
        None: The function directly displays histograms of sentiment scores for each cuisine type.
    """
    score_column = 'Compound Score (VADER)' if vader else 'Polarity Score (TextBlob)'

    cuisine_types = data['Cuisines_List'].unique()

    num_rows = (len(cuisine_types) // 6) + (len(cuisine_types) % 6 != 0)

    fig, axes = plt.subplots(num_rows, 6, figsize = (24, num_rows * 4))

    axes = axes.flatten()

    for idx, cuisine in enumerate(cuisine_types):
        cuisine_data = data[data['Cuisines_List'] == cuisine]
        
        sns.histplot(cuisine_data[score_column], kde = True, ax = axes[idx], color = (141/255, 160/255, 203/255))
        
        axes[idx].set_title(f'{score_column} for {cuisine}', fontsize = 12)
        axes[idx].set_xlabel(score_column, fontsize = 10)
        axes[idx].set_ylabel('Frequency', fontsize = 10)

    for idx in range(len(cuisine_types), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def plot_histogram_TextBlob(dataframe):
    """
    Plots histograms of polarity and subjectivity scores from TextBlob sentiment analysis.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing the polarity and subjectivity scores from TextBlob sentiment analysis.

    Returns:
        None: The function directly displays histograms showing the distributions of polarity and subjectivity scores.
    """
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ('Polarity Distribution', 'Subjectivity Distribution'))

    fig.add_trace(go.Histogram(x = dataframe['Polarity Score (TextBlob)'], nbinsx = 20, name = 'Polarity',
        marker_color = 'rgb(141, 160, 203)'), row = 1, col = 1)

    fig.add_trace(go.Histogram(x = dataframe['Subjectivity Score (TextBlob)'], nbinsx = 20, name = 'Subjectivity',
        marker_color = 'rgb(141, 160, 203)'), row = 1, col = 2)

    fig.update_layout(title = 'Polarity and Subjectivity Distributions', showlegend = False, bargap = 0.1, height = 500)

    fig.show()

def plot_normalized_distributions(Rating_Scaled, Compound_Score_Vader_Scaled, Polarity_Score_TextBlob_Scaled):
    """
    Plots the normalized distributions of scaled Rating, Compound Score (VADER), and Polarity Score (TextBlob) using histograms
    and kernel density estimates (KDE).

    Parameters:
        Rating_Scaled (pd.Series or array-like): The scaled rating values to be plotted.
        Compound_Score_Vader_Scaled (pd.Series or array-like): The scaled compound scores from VADER sentiment analysis to be plotted.
        Polarity_Score_TextBlob_Scaled (pd.Series or array-like): The scaled polarity scores from TextBlob sentiment analysis to be plotted.

    Returns:
        None: The function displays the histograms and KDEs for each of the provided datasets.
    """
    sns.set(style = "whitegrid")

    fig, axes = plt.subplots(1, 3, figsize = (18, 6))

    sns.histplot(Rating_Scaled, kde = True, bins = 20, color = 'rgb(141, 160, 203)', ax = axes[0], legend = False)
    axes[0].set_title("Rating")
    axes[0].set_ylim(0, 5000)  
    axes[0].lines[0].set_color('black')

    sns.histplot(Compound_Score_Vader_Scaled, kde = True, bins = 20, color = 'rgb(141, 160, 203)', ax = axes[1], legend = False)
    axes[1].set_title("Compound Score (VADER)")
    axes[1].set_ylim(0, 5000)  
    axes[1].lines[0].set_color('black')

    sns.histplot(Polarity_Score_TextBlob_Scaled, kde = True, bins = 20, color = 'rgb(141, 160, 203)', ax = axes[2], legend = False)
    axes[2].set_title("Polarity Score (TextBlob)")
    axes[2].set_ylim(0, 5000)  
    axes[2].lines[0].set_color('black')

    plt.tight_layout()

    plt.show()

def plot_regression(Compound_Score_Vader_Scaled, Polarity_Score_TextBlob_Scaled):
    """
    Plots a regression scatter plot to visualize the relationship between scaled Compound Score (VADER) and scaled Polarity Score (TextBlob).
    It also includes a trendline (Ordinary Least Squares regression).

    Parameters:
        Compound_Score_Vader_Scaled (array-like): The scaled compound scores from VADER sentiment analysis.
        Polarity_Score_TextBlob_Scaled (array-like): The scaled polarity scores from TextBlob sentiment analysis.

    Returns:
        None: Displays a scatter plot with a regression trendline.
    """
    Compound_Score_Vader_Scaled = Compound_Score_Vader_Scaled.flatten()
    Polarity_Score_TextBlob_Scaled = Polarity_Score_TextBlob_Scaled.flatten()
    
    data_to_plot = pd.DataFrame({'Compound_Score_Vader_Scaled': Compound_Score_Vader_Scaled, 'Polarity_Score_TextBlob_Scaled': Polarity_Score_TextBlob_Scaled})
    
    fig = px.scatter(data_to_plot, x = 'Compound_Score_Vader_Scaled', y = 'Polarity_Score_TextBlob_Scaled', trendline = "ols", labels = {"Compound_Score_Vader_Scaled": "Compound Score (VADER)", "Polarity_Score_TextBlob_Scaled": "Polarity Score (TextBlob)"}, title = "Regression Plot: Compound Score (VADER) vs. Polarity Score (TextBlob)")

    fig.update_traces(marker = dict(color = 'rgb(141, 160, 203)'))  
    fig.update_traces(line = dict(color = 'black'), selector = dict(mode = 'lines'))

    fig.show()

def heatmap_topic_modelling(data, model, vectorizer):
    """
    Generates a heatmap to visualize the correlation between the dominant topics in a topic modeling model (e.g., LDA, NMF) and 
    the vectorizer used (e.g., TF-IDF, CountVectorizer). The heatmap is based on the correlation of dummy variables for the dominant topics.

    Parameters:
        data (pd.DataFrame): A DataFrame containing the dataset with a column for the dominant topics.
        model (str): The name of the topic modeling model (e.g., 'LDA', 'NMF').
        vectorizer (str): The name of the vectorizer used (e.g., 'tfidf', 'count').

    Returns:
        heatmap (plotly.graph_objects.Heatmap): A Plotly heatmap object visualizing the correlation matrix of the dominant topics.
    """
    correlation_matrix = pd.get_dummies(data[f"Dominant_Topic_{model}_{vectorizer.upper()}"], prefix = "Topic").corr()

    mask = np.tril(np.ones_like(correlation_matrix, dtype = bool), k = -1) 
    correlation_matrix[mask] = np.nan

    heatmap = go.Heatmap(z = correlation_matrix.values, x = correlation_matrix.columns, y = correlation_matrix.columns, colorscale = 'Blues', colorbar = dict(title = "Correlation", tickvals = [-1, 0, 1]), text = np.where(np.isnan(correlation_matrix.values), '', correlation_matrix.round(2)), texttemplate = "%{text}", coloraxis = "coloraxis")
    
    return heatmap

# ==========================================================================================================================================
# Text Processing
# ==========================================================================================================================================

def clean_dataframe(dataframe, column_name, remove_emojis = True, remove_punctuation = True, remove_case = True, remove_stopwords = True, sentence_vectorizer = False):
    """
    Cleans a specified text column in a DataFrame by performing multiple preprocessing steps such as removing emojis, 
    punctuation, stopwords, and converting digits to words. Optionally, text can be lowercased and tokenized for sentence vectorization.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the text data to be cleaned.
        column_name (str): The name of the column in the DataFrame that contains the text data to be cleaned.
        remove_emojis (bool): Whether to remove emojis from the text. Default is True.
        remove_punctuation (bool): Whether to remove punctuation from the text. Default is True.
        remove_case (bool): Whether to convert the text to lowercase. Default is True.
        remove_stopwords (bool): Whether to remove common stopwords from the text. Default is True.
        sentence_vectorizer (bool): Whether to tokenize the text by sentences using PunktSentenceTokenizer or words using word_tokenize.
                                    Default is False (word_tokenize).

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'Review_cleaned' column containing the cleaned text data.
    """
    newline_pattern = r"(\n)"
    hashtags_pattern = r"([#@])"
    url_pattern = r"(?:\w+:\/\/)?(?:www\.)?[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?"
    
    punctuation_translator = str.maketrans('', '', string.punctuation)
    
    stopwords_set = set(stopwords.words("english"))
    additional_stopwords = {'the'}
    stopwords_set.update(additional_stopwords)

    negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor'}

    def convert_digit_to_word(text):
        """
        Converts numbers, including decimals and fractions, into words.
        
        This function processes decimal numbers (e.g., 3.5 -> 'three point five'), fractions (e.g., 1/2 -> 'one over two'), 
        and general integers into their word equivalents.
        
        Parameters:
            text (str): The text containing the numbers to be converted.
        
        Returns:
            str: The text with numbers converted to words.
        """
        # Match decimal numbers (e.g., 3.5)
        def convert_decimal(match):
            parts = match.group().split('.')
            before_decimal = num2words(parts[0])
            after_decimal = num2words(parts[1])
            return f"{before_decimal} point {after_decimal}"

        # Match fractions (e.g., 1/2)
        def convert_fraction(match):
            parts = match.group().split('/')
            numerator = num2words(parts[0])
            denominator = num2words(parts[1])
            return f"{numerator} over {denominator}"

        # Convert ordinal numbers (e.g., 1st, 2nd, 3rd) to words
        def convert_ordinal(match):
            number = int(match.group()[:-2])  # Remove 'st', 'nd', 'rd', 'th' and convert the number
            return num2words(number) + match.group()[-2:]  # Keep the ordinal suffix ('st', 'nd', etc.)

        text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', convert_ordinal, text)
        text = re.sub(r'\b\d+\.\d+\b', convert_decimal, text)  
        text = re.sub(r'\b\d+/\d+\b', convert_fraction, text)  
        text = re.sub(r'\b\d+\b', lambda x: num2words(x.group()), text)
        
        return text

    def clean_text(raw_text, remove_emojis, remove_punctuation, remove_case, remove_stopwords, sentence_vectorizer):
        """
        Cleans a raw text string by applying various text preprocessing steps based on the provided parameters.
        
        Parameters:
            raw_text (str): The raw text to be cleaned.
            remove_emojis (bool): Whether to remove emojis from the text.
            remove_punctuation (bool): Whether to remove punctuation from the text.
            remove_case (bool): Whether to convert text to lowercase.
            remove_stopwords (bool): Whether to remove common stopwords.
            sentence_vectorizer (bool): Whether to tokenize the text by sentences or words.
        
        Returns:
            list: A list of tokens obtained after cleaning and tokenizing the raw text.
        """
        clean_text = raw_text
        
        if remove_emojis:
            clean_text = emoji.replace_emoji(clean_text, replace = "")
        
        clean_text = re.sub(hashtags_pattern, "", clean_text)
        clean_text = re.sub(newline_pattern, " ", clean_text)
        clean_text = re.sub(url_pattern, "", clean_text)
        
        if remove_punctuation:
            clean_text = clean_text.translate(punctuation_translator)
        
        if remove_case:
            clean_text = clean_text.lower()

        clean_text = re.sub(r'\b[a-zA-Z]\b', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        clean_text = re.sub(r'\b\d+\b', lambda x: convert_digit_to_word(x.group()), clean_text)

        if sentence_vectorizer:
            tokenizer = PunktSentenceTokenizer()
            tokens = tokenizer.tokenize(clean_text)

        else:
            tokens = word_tokenize(clean_text)
        
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stopwords_set or word in negation_words]

        return tokens

    dataframe['Review_cleaned'] = dataframe[column_name].apply(lambda x: clean_text(x, remove_emojis, remove_punctuation, remove_case, remove_stopwords, sentence_vectorizer))
    
    return dataframe

nlp = spacy.load("en_core_web_sm") # If we use the pos_tag from nlkt, it recognizes all words ending in 'ing' as VBG (gerund), however, some of them are nouns, like 'evening'.

def lemmatize_tokens(row_of_tokens):
    """
    Lemmatizes a list of tokens (words) and retrieves their corresponding part-of-speech tags.

    This function takes a list of tokens (words) as input, joins them into a single string, and processes them with 
    a spaCy NLP pipeline to obtain their lemmatized forms and part-of-speech tags.

    Parameters:
        row_of_tokens (list): A list of tokens (words) that need to be lemmatized.

    Returns:
        tuple: A tuple containing two lists:
            - lemmas (list): A list of lemmatized words.
            - pos_tags (list): A list of part-of-speech tags corresponding to the lemmatized words.
    """
    doc = nlp(" ".join(row_of_tokens))
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    return lemmas, pos_tags

# ==========================================================================================================================================
# Vectorization
# ==========================================================================================================================================

def similarity(model, input_word):
    """
    Computes the most and least similar words to a given word using a Word2Vec model. This function takes a word as 
    input and uses the provided Word2Vec model to find the most similar and least similar words based on cosine similarity. 
    It returns the most and least similar words along with their similarity scores.

    Parameters:
        model (gensim.models.Word2Vec): A trained Word2Vec model.
        input_word (str): The input word for which the most and least similar words will be found.

    Returns:
        tuple: A tuple containing:
            - most_similar_word (str): The most similar word to the input word.
            - most_similar_score (float): The cosine similarity score for the most similar word.
            - least_similar_word (str): The least similar word to the input word.
            - least_similar_score (float): The cosine similarity score for the least similar word.
    """
    most_similar = model.wv.most_similar(positive = [input_word], topn = 1)
    least_similar = model.wv.most_similar(negative = [input_word], topn = 1)

    most_similar_word, most_similar_score = most_similar[0]
    least_similar_word, least_similar_score = least_similar[0]
    
    return most_similar_word, most_similar_score, least_similar_word, least_similar_score

def prediction(model, input_word):
    """
    Predicts the most and least similar words to a given word using a Word2Vec model's output prediction. This function takes 
    a word as input and uses the Word2Vec model's `predict_output_word` method to find the top 10 most likely words based 
    on the input word. It then sorts the predicted words by their similarity score and returns the most and least similar 
    words along with their respective similarity scores.

    Parameters:
        model (gensim.models.Word2Vec): A trained Word2Vec model.
        input_word (str): The input word for which the most and least similar words will be predicted.

    Returns:
        tuple: A tuple containing:
            - most_similar_word (str): The most similar word to the input word.
            - most_similar_score (float): The similarity score for the most similar word.
            - least_similar_word (str): The least similar word to the input word.
            - least_similar_score (float): The similarity score for the least similar word.
"""
    predicted_words = model.predict_output_word([input_word], topn = 10)  
    predicted_words_sorted = sorted(predicted_words, key = lambda x: x[1], reverse = True)

    most_similar_word, most_similar_score = predicted_words_sorted[0]
    least_similar_word, least_similar_score = predicted_words_sorted[-1]

    return most_similar_word, most_similar_score, least_similar_word, least_similar_score

def text_generator(model, tokens_sequence, number_to_generate, random_nr = False, random_nr_max = 5):
    """
    Generates new words based on a sequence of input tokens using a trained Word2Vec model. This function takes a sequence 
    of tokens (words) and generates a specified number of new words based on the trained Word2Vec model. The generated words 
    are appended to the original sequence. The selection of the next word can either be deterministic (always selecting the top word) 
    or random (randomly choosing one of the top N predicted words).

    Parameters:
        model (gensim.models.Word2Vec): A trained Word2Vec model used to predict the next word in the sequence.
        tokens_sequence (list): A list of tokens (words) that will form the basis for the word generation.
        number_to_generate (int): The number of new words to generate and append to the tokens sequence.
        random_nr (bool, optional): Whether to choose a word randomly from the top N predicted words (default is False).
        random_nr_max (int, optional): The maximum number of top predicted words to choose from when `random_nr` is True (default is 5).

    Returns:
        list: The updated sequence of tokens with the newly generated words appended.
    """
    sequence = tokens_sequence

    for _ in range(number_to_generate):
        if len(sequence) == 0:
            break
  
        if random_nr:
            j = random.randrange(random_nr_max)
        else:
            j = 0  

        predicted_words = model.predict_output_word(sequence, topn = random_nr_max)
        
        new_word = predicted_words[j][0] 
        sequence.append(new_word)

    return sequence

def embedding_function(model, vocabulary):
    """
    Generates word embeddings using a trained Word2Vec model and applies PCA to reduce the dimensionality 
    of word vectors to 2 dimensions for visualization. This function takes a list of words (vocabulary) and uses a pre-trained 
    Word2Vec model to retrieve the corresponding word embeddings (vectors). The function then applies Principal Component Analysis 
    (PCA) to reduce the word vectors from high-dimensional space (typically 100-300 dimensions) to 2D space for easier visualization. 
    The resulting 2D embeddings are returned along with the original vectors.

    Parameters:
        model (gensim.models.Word2Vec): A trained Word2Vec model used to obtain word vectors for each word in the vocabulary.
        vocabulary (list): A list of words (vocabulary) for which embeddings are to be extracted and reduced to 2D.

    Returns:
        pd.DataFrame: A DataFrame containing the original words, their high-dimensional vectors, and the 2D PCA components.
    """
    pca_2 = PCA(n_components = 2)
    matrix = np.array([model.wv[word].tolist() for word in vocabulary])
    pca_result = pca_2.fit_transform(matrix)
    
    result = pd.DataFrame({"Words":vocabulary, "Vectors": matrix.tolist(), "PCA_dim_1": pca_result[:,0],"PCA_dim_2": pca_result[:,1]})
    
    return result

def sentence_vectorizer(token_list, model, vector_size):
    """
    Converts a list of tokens (words) into a single sentence vector by averaging the word vectors. This function takes a 
    list of tokens (words) from a sentence, retrieves the word vectors (embeddings) for each token from a pre-trained word 
    embedding model (e.g., Word2Vec), and computes the average of those vectors to generate a single vector representation 
    of the sentence. If a word is not found in the vocabulary of the model, it is ignored.

    Parameters:
        token_list (list): A list of tokens (words) in the sentence.
        model (gensim.models.KeyedVectors): A pre-trained word embedding model (e.g., Word2Vec) used to retrieve word vectors 
                                            for the tokens.
        vector_size (int): The size of the word vectors used by the model. This value determines the dimensionality of the 
                         resulting sentence vector.

    Returns:
        numpy.ndarray: A vector representing the sentence, which is the average of the word vectors for all tokens in the 
                     input token list.
    """
    sentence_vector = np.zeros(vector_size, dtype = "f")
    sentence_length = len(token_list)

    for token in token_list:
        try:
            token_vector = model[token]
            sentence_vector = sentence_vector + token_vector
        except KeyError:
            sentence_length = sentence_length - 1

    sentence_vector = np.divide(sentence_vector, sentence_length)

    return sentence_vector

def manual_loading(file_path):
    """
    Loads a pre-trained GloVe word embeddings model from a file into a dictionary. This function reads a GloVe file where 
    each line contains a word followed by its corresponding word vector (embedding). The word vectors are stored in a dictionary 
    where the keys are the words and the values are the corresponding vectors.

    Parameters:
        file_path (str): The path to the GloVe file. The file should have words in the first column and their corresponding 
        word vectors in subsequent columns.

    Returns:
        dict: A dictionary where the keys are words and the values are numpy arrays representing the word vectors (embeddings).
    """
    glove_model = {}
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype = 'float32')
            glove_model[word] = vector
    return glove_model

def glove_vectorization(lemmas, glove_model):
    """
    Converts a list of lemmas (tokens) into a single word vector by averaging the corresponding word vectors from a 
    pre-trained GloVe model. This function takes a list of lemmatized words (lemmas) and finds the corresponding GloVe 
    word vectors for each lemma. If a lemma is not found in the GloVe model, a zero vector is used for that lemma. 
    The final vector is the average of the word vectors of the lemmas in the input list.

    Parameters:
        lemmas (list): A list of lemmatized words (tokens) for which to compute the word vector.
        glove_model (dict): A pre-trained GloVe model, represented as a dictionary where the keys are words and the values 
                            are the corresponding word vectors (numpy arrays).

    Returns:
        numpy.ndarray: A 50-dimensional word vector representing the average of the GloVe vectors of the lemmas.
                     If no lemmas are found in the GloVe model, a zero vector is returned.
    """
    word_vectors = []

    for lemma in lemmas:
        if lemma in glove_model:
            word_vectors.append(glove_model[lemma])
        else:
            word_vectors.append(np.zeros((50,))) 

    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis = 0)
    else:
        return np.zeros((50,))

# ==========================================================================================================================================
# Multi Label Classification
# ==========================================================================================================================================

def calculate_metrics(model, x_val, y_val):
    """
    Calculates and returns evaluation metrics (F1 Score, Precision, Recall) for a given model based on its predictions 
    on the validation set. This function computes three key classification evaluation metrics for a model's predictions:
        - **F1 Score**: The harmonic mean of Precision and Recall, used to balance the trade-off between them.
        - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
        - **Recall**: The ratio of correctly predicted positive observations to all the actual positives.

    Parameters:
        model: The trained classification model that will make predictions on the validation data.
        x_val (array-like): The feature set used for validation (input data).
        y_val (array-like): The true labels for the validation set (target values).

    Returns:
        dict: A dictionary containing the following metrics:
            - "F1 Score": The weighted F1 Score.
            - "Precision": The weighted Precision.
            - "Recall": The weighted Recall.
    """
    y_pred = model.predict(x_val)
    metrics = {
        "F1 Score": f1_score(y_val, y_pred, average = 'weighted'),
        "Precision": precision_score(y_val, y_pred, average = 'weighted'),
        "Recall": recall_score(y_val, y_pred, average = 'weighted')}
    return metrics

# ==========================================================================================================================================
# Sentiment Analysis
# ==========================================================================================================================================

Vader_Sentiment_Analyzer = SentimentIntensityAnalyzer()

def Vader_Wrapper(sentences, mean_sentence_polarity_vader = True):
    """
    Analyzes the sentiment of the provided sentences using the VADER sentiment analysis tool. This function utilizes 
    the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool to compute sentiment scores for 
    a given text or a list of sentences. It calculates the overall compound sentiment score for the full text, and 
    optionally computes the mean compound score of individual sentences if the `mean_sentence_polarity_vader` flag is set to True.

    Parameters:
        sentences (str or list of str): The text or a list of sentences to analyze. If a list is provided, the sentiment 
                                        analysis will be applied to each sentence individually.
        mean_sentence_polarity_vader (bool, optional): If True, computes the mean compound score for the individual sentences 
                                                     in the list. Defaults to True. If False, only the overall compound 
                                                     score for the entire text is computed.

    Returns:
        tuple: A tuple containing two values:
            - `overall_compound` (float): The overall compound sentiment score for the entire text.
            - `mean_compound` (float): The mean compound sentiment score of individual sentences if the `mean_sentence_polarity_vader` 
                                     flag is True. Otherwise, it returns the same value as `overall_compound`.
    """
    if isinstance(sentences, list):
        full_text = ' '.join(sentences)
    else:
        full_text = sentences

    overall_polarity_scores = Vader_Sentiment_Analyzer.polarity_scores(full_text)
    overall_compound = overall_polarity_scores["compound"]

    if mean_sentence_polarity_vader and isinstance(sentences, list):
        compound_scores = []
        for sentence in sentences:
            polarity_scores = Vader_Sentiment_Analyzer.polarity_scores(sentence)
            compound_scores.append(polarity_scores["compound"])
        mean_compound = np.mean(compound_scores)
    else:
        mean_compound = overall_compound

    return overall_compound, mean_compound

def Apply_Vader(dataframe, column = 'Review_cleaned', mean_sentence = True):
    """
    Applies VADER sentiment analysis to a specified column in a dataframe and adds sentiment scores. This function takes 
    a dataframe and applies the VADER sentiment analysis tool to the text data in the specified column. It computes the 
    compound sentiment scores for each review and, if requested, the mean compound score for individual sentences within 
    the review. The results are added as new columns in the dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The dataframe containing the text data.
        column (str, optional): The name of the column in the dataframe that contains the text data to be analyzed.
                                Defaults to 'Review_cleaned'.
        mean_sentence (bool, optional): If True, computes the mean compound sentiment score of individual sentences
                                        within each review. If False, only the overall compound score for the entire review
                                        is calculated. Defaults to True.

    Returns:
        pandas.DataFrame: The original dataframe with two additional columns:
            - 'Compound Score (VADER)': The overall compound sentiment score for each review.
            - 'Mean Compound Score (VADER)': The mean compound sentiment score of individual sentences in the review (if `mean_sentence` is True).
    """
    results = dataframe[column].apply(lambda x: pd.Series(Vader_Wrapper(x, mean_sentence)))
    
    dataframe['Compound Score (VADER)'] = results[0]         
    dataframe['Mean Compound Score (VADER)'] = results[1] 
    
    return dataframe

def extract_scores(dataframe, review_column):
    """
    Extracts sentiment scores from text reviews using the VADER sentiment analysis tool and adds them as new columns in the dataframe.
    This function uses the VADER Sentiment Analyzer to compute the positive, negative, and neutral sentiment scores for each review
    in the specified column. The function then adds these scores as new columns ('Positive', 'Negative', 'Neutral') to the dataframe.

    Parameters:
        dataframe (pandas.DataFrame): The dataframe containing the reviews.
        review_column (str): The name of the column in the dataframe that contains the text reviews to be analyzed.

    Returns:
        pandas.DataFrame: The original dataframe with three additional columns:
            - 'Positive': The positive sentiment score for each review.
            - 'Negative': The negative sentiment score for each review.
            - 'Neutral': The neutral sentiment score for each review.
    """
    scores = dataframe[review_column].apply(lambda x: Vader_Sentiment_Analyzer.polarity_scores(x))
    dataframe['Positive'] = scores.apply(lambda x: x['pos'])
    dataframe['Negative'] = scores.apply(lambda x: x['neg'])
    dataframe['Neutral'] = scores.apply(lambda x: x['neu'])
    return dataframe

def TextBlop_Wrapper(review_text):
    """
    Analyzes the sentiment of a given text (or list of texts) using the TextBlob library, extracting the polarity and subjectivity.
    This function takes a single review or a list of reviews, joins them if needed, and then calculates the sentiment polarity 
    and subjectivity using TextBlob. Polarity represents the sentiment of the text, ranging from -1 (negative) to 1 (positive), 
    while subjectivity represents how subjective the text is, ranging from 0 (objective) to 1 (subjective).

    Parameters:
        review_text (str or list of str): The text or list of texts (reviews) to analyze. If a list of reviews is provided, 
                                         they are joined into a single string.

    Returns:
        tuple: A tuple containing two values:
            - polarity (float): The polarity score of the sentiment. Ranges from -1 (negative) to 1 (positive).
            - subjectivity (float): The subjectivity score of the sentiment. Ranges from 0 (objective) to 1 (subjective).
    """
    if isinstance(review_text, list):
        review_text = ' '.join(review_text)
    
    sentiment = TextBlob(review_text).sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    
    return polarity, subjectivity

def Apply_TextBlob(dataframe, column = 'Review_cleaned'):
    """
    Applies TextBlob sentiment analysis to a specified column in a DataFrame, extracting polarity and subjectivity scores.
    This function analyzes the sentiment of each entry in the specified column of the DataFrame using the TextBlob library. 
    It computes the polarity and subjectivity scores for each entry, then adds these as new columns to the DataFrame. 
    The polarity score indicates the sentiment of the text (from -1 for negative to 1 for positive), and the subjectivity 
    score measures the degree to which the text is subjective (from 0 for objective to 1 for subjective).

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the text data.
        column (str): The name of the column in the DataFrame to which sentiment analysis will be applied. Default is 'Review_cleaned'.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns added:
            - 'Polarity Score (TextBlob)': The polarity score of the sentiment for each entry in the specified column.
            - 'Subjectivity Score (TextBlob)': The subjectivity score of the sentiment for each entry in the specified column.
    """
    results = dataframe[column].apply(lambda x: pd.Series(TextBlop_Wrapper(x), index = ['Polarity Score (TextBlob)', 'Subjectivity Score (TextBlob)']))
    
    dataframe[['Polarity Score (TextBlob)', 'Subjectivity Score (TextBlob)']] = results
    
    return dataframe

def Sentiment_Analysis_Metrics(Rating_Scaled, Compound_Score_Vader_Scaled, Polarity_Score_TextBlob_Scaled, epsilon = 1):
    """
    Calculates various sentiment analysis metrics to evaluate the performance of sentiment scores (VADER and TextBlob) 
    against the actual ratings (scaled). This function computes several evaluation metrics, including Mean Squared Error (MSE), 
    Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Root Mean Absolute Error (RMAE), Mean Absolute Percentage Error (MAPE), 
    and Pearson correlation for both VADER and TextBlob sentiment scores.

    Parameters:
        Rating_Scaled (np.ndarray or pd.Series): The actual ratings, which are expected to be scaled.
        Compound_Score_Vader_Scaled (np.ndarray or pd.Series): The VADER sentiment analysis scores, scaled.
        Polarity_Score_TextBlob_Scaled (np.ndarray or pd.Series): The TextBlob sentiment analysis scores, scaled.
        epsilon (float): A small constant added to avoid division by zero when calculating MAPE (default is 1).

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics for both VADER and TextBlob sentiment analysis methods.
            - Rows correspond to the metrics: MSE, RMSE, MAE, RMAE, MAPE, and Pearson correlation.
            - Columns correspond to the evaluation metrics for VADER and TextBlob.
    """
    adjust_1 = Rating_Scaled.reshape(-1) + epsilon
    adjust_2 = Compound_Score_Vader_Scaled.reshape(-1) + epsilon
    adjust_3 = Polarity_Score_TextBlob_Scaled.reshape(-1) + epsilon
    
    MSE_Compound_Score_VADER = mean_squared_error(Rating_Scaled.reshape(-1), Compound_Score_Vader_Scaled.reshape(-1))
    MSE_Polarity_Score_TextBlob = mean_squared_error(Rating_Scaled.reshape(-1), Polarity_Score_TextBlob_Scaled.reshape(-1))

    RMSE_Compound_Score_VADER = mean_squared_error(Rating_Scaled.reshape(-1), Compound_Score_Vader_Scaled.reshape(-1), squared = False)
    RMSE_Polarity_Score_TextBlob = mean_squared_error(Rating_Scaled.reshape(-1), Polarity_Score_TextBlob_Scaled.reshape(-1), squared = False)

    MAE_Compound_Score_VADER = mean_absolute_error(Rating_Scaled.reshape(-1), Compound_Score_Vader_Scaled.reshape(-1))
    MAE_Polarity_Score_TextBlob = mean_absolute_error(Rating_Scaled.reshape(-1), Polarity_Score_TextBlob_Scaled.reshape(-1))

    RMAE_Compound_Score_VADER = MAE_Compound_Score_VADER ** 0.5
    RMAE_Polarity_Score_TextBlob = MAE_Polarity_Score_TextBlob ** 0.5

    MAPE_Compound_Score_VADER = (abs(adjust_1 - adjust_2) / adjust_1).mean() * 100
    MAPE_Polarity_Score_TextBlob = (abs(adjust_1 - adjust_3) / adjust_1).mean() * 100

    correlation_VADER, _ = pearsonr(Rating_Scaled.reshape(-1), Compound_Score_Vader_Scaled.reshape(-1))
    correlation_TextBlob, _ = pearsonr(Rating_Scaled.reshape(-1), Polarity_Score_TextBlob_Scaled.reshape(-1))

    metrics = pd.DataFrame({"VADER": [MSE_Compound_Score_VADER, RMSE_Compound_Score_VADER, MAE_Compound_Score_VADER, RMAE_Compound_Score_VADER, MAPE_Compound_Score_VADER, correlation_VADER], "TextBlob": [MSE_Polarity_Score_TextBlob, RMSE_Polarity_Score_TextBlob, MAE_Polarity_Score_TextBlob, RMAE_Polarity_Score_TextBlob, MAPE_Polarity_Score_TextBlob, correlation_TextBlob]}, index = ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "Root Mean Absolute Error (RMAE)", "Mean Absolute Percentage Error (MAPE)", "Pearson Correlation"])

    return metrics

# ==========================================================================================================================================
# Topic Modelling
# ==========================================================================================================================================

def perform_topic_modelling(text_column, vectorizer, vocab = None, max_k = 10, model_type = 'LSA', n_top_words = 10):
    """
    Generates a topic modeling solution using LSA or LDA.
    
    Parameters:
        text_column (pd.Series or list): The text data to model.
        vectorizer (str): The vectorization method ('bow' or 'tfidf').
        vocab (list or None): Vocabulary to constrain the vectorizer.
        max_k (int): Number of topics.
        model_type (str): Topic modeling technique ('LSA' or 'LDA').
        n_top_words (int): Number of top words per topic.
    
    Returns:
        topic_matrix (np.ndarray): Document-topic matrix.
        model (object): Fitted LSA or LDA model.
        vectorizer_model (object): Fitted vectorizer model.
        topic_top_words (list): List of top words for each topic.
    """

    if vectorizer == 'bow':
        model_vectorization = CountVectorizer(stop_words = 'english', vocabulary = vocab)

    elif vectorizer == 'tfidf':
        model_vectorization = TfidfVectorizer(stop_words = 'english', vocabulary = vocab)
    else:
        raise ValueError("Unsupported vectorizer. Choose 'bow' or 'tfidf'.")
    
    vectorized_representation = model_vectorization.fit_transform(text_column)
    
    if model_type == 'LSA':
        model = TruncatedSVD(n_components = max_k, random_state = 42)
        topic_matrix = model.fit_transform(vectorized_representation)

    elif model_type == 'LDA':
        model = LatentDirichletAllocation(n_components = max_k, random_state = 42)
        topic_matrix = model.fit_transform(vectorized_representation)

    else:
        raise ValueError("Unsupported model type. Choose 'LSA' or 'LDA'.")
    
    feature_names = model_vectorization.get_feature_names_out()
    topic_top_words = []

    for topic_idx, topic in enumerate(model.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topic_top_words.append(', '.join(top_words))
    
    return topic_matrix, model, model_vectorization, topic_top_words

def run_topic_modelling(data, vectorization_methods, models, max_k, n_top_words, text_column = 'Lemmas_Treated'):
    """
    Performs topic modeling using specified vectorization methods and models, computes coherence scores.

    Parameters:
        data (pd.DataFrame): The dataset containing the text data.
        vectorization_methods (list): List of vectorization methods ('bow', 'tfidf').
        models (list): List of topic modeling techniques ('LSA', 'LDA').
        max_k (int): Number of topics to extract.
        n_top_words (int): Number of top words per topic.
        text_column (str): Column name containing the text data.

    Returns:
        dict: A dictionary with results and coherence scores for each combination.
    """

    results = {}
    texts = [text.split() for text in data[text_column]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    for vectorizer in vectorization_methods:
        for model_type in models:
            print(f"Running {model_type} with {vectorizer.upper()}...\n")
            
            matrix, model, vectorizer_model, topic_words = perform_topic_modelling(data[text_column], vectorizer = vectorizer, vocab = None, max_k = max_k, model_type = model_type, n_top_words = n_top_words)
            
            dominant_topic_column = f'Dominant_Topic_{model_type}_{vectorizer.upper()}'
            topic_words_column = f'Topic_Top_Words_{model_type}_{vectorizer.upper()}'
            
            data[dominant_topic_column] = matrix.argmax(axis = 1)
            topic_mapping = {i: words for i, words in enumerate(topic_words)}
            data[topic_words_column] = data[dominant_topic_column].map(topic_mapping)
            
            coherence_model = CoherenceModel(topics = [topic.split(', ') for topic in topic_words], dictionary = dictionary, corpus = corpus, texts = texts, coherence = 'c_v')
            coherence_score = coherence_model.get_coherence()
            
            print(f"Top words for {model_type} with {vectorizer.upper()} topics:")
            for topic_idx, words in topic_mapping.items():
                print(f"Topic {topic_idx + 1}: {words}")
            
            print(f"\nCoherence Score for {model_type} with {vectorizer.upper()}: {coherence_score:.4f}\n")
            
            results[f"{model_type}_{vectorizer.upper()}"] = {'Topic Mapping': topic_mapping, 'Coherence Score': coherence_score}
    
    return results

# ==========================================================================================================================================
# Co-occurrence Analysis
# ==========================================================================================================================================

dishes = [
    'Aalishaan Chicken',
    'Achaari Paneer Tikka',
    'Achari Chicken',
    'Achari Paneer Tikka',
    'Achari Paneer Tikka\nChole Kulche\nGalowti Kabab\nShikanji',
    'Afghan Kebab',
    'Afghani Chicken',
    'Agnolotti Pasta',
    'Ajwaini Paneer',
    'Alfredo Chicken Pasta',
    'Alfredo Pasta',
    'All Mutton',
    'Aloo Paratha',
    'Aloo Parathas',
    'Amritsari Fish',
    'Andhra Chicken Tikka',
    'Andhra Pepper Chicken Pizza',
    'Andhra Veg Biryani',
    'Apollo Fish',
    'Apple Iced Tea',
    'Awadhi Biryani',
    'Awadhi Chicken Biryani',
    'Bamboo Biryani',
    'Banana Milkshake',
    'Barbeque Chicken',
    'Barfi Ice Cream',
    'Basil Pesto Pasta',
    'Bauli Paneer Tikka',
    'Bengali Fish Curry',
    'Bhuna Mutton',
    'Bhutta Kebab',
    'Bihari Boti Kebab',
    'Butter Chicken',
    'Butter Naan',
    'Butter Paneer',
    'Butter Paneer Masala',
    'Butter Paratha',
    'Cajun Chicken',
    'Caramel Custard',
    'Cheese Burger',
    'Cheese Garlic Bread',
    'Cheese Naan',
    'Cheese Paratha',
    'Cheese Pizza',
    'Chicken',
    'Chicken Alfredo Pasta',
    'Chicken Biryani',
    'Chicken Burger',
    'Chicken Curry',
    'Chicken Kebab',
    'Chicken Manchurian',
    'Chicken Noodles',
    'Chicken Pasta',
    'Chicken Pizza',
    'Chicken Sandwich',
    'Chicken Shawarma',
    'Chicken Soup',
    'Chicken Tandoori',
    'Chicken Tikka',
    'Chicken Wings',
    'Choco Lava Cake',
    'Choco Milkshake',
    'Chocolate Brownie',
    'Chocolate Cake',
    'Chocolate Ice Cream',
    'Chocolate Milkshake',
    'Chole Bhature',
    'Corn Soup',
    'Creamy Alfredo Pasta',
    'Crispy Chicken',
    'Dal Makhani',
    'Egg Biryani',
    'Egg Curry',
    'Egg Fried Rice',
    'Fish Curry',
    'Garlic Naan',
    'Garlic Paratha',
    'Green Salad',
    'Grilled Chicken',
    'Gulab Jamun',
    'Ice Cream Sundae',
    'Indian Chicken Curry',
    'Italian Pasta',
    'Kadhai Chicken',
    'Kadhai Paneer',
    'Kulfi Falooda',
    'Lamb Biryani',
    'Lamb Curry',
    'Mango Ice Cream',
    'Masala Chai',
    'Masala Dosa',
    'Mixed Veg Curry',
    'Mutton Biryani',
    'Mutton Curry',
    'Paneer Butter Masala',
    'Paneer Tikka',
    'Pesto Pasta',
    'Plain Naan',
    'Plain Paratha',
    'Prawns Curry',
    'Prawn Biryani',
    'Shahi Paneer',
    'Shawarma Roll',
    'Spinach Corn Sandwich',
    'Sweet Corn Soup',
    'Tandoori Chicken',
    'Tandoori Fish',
    'Veg Biryani',
    'Veg Fried Rice',
    'Veg Hakka Noodles',
    'Veg Manchurian',
    'Veg Pasta',
    'Veg Pizza']

def find_dishes(review):
    """
    Identifies dishes mentioned in a review from a predefined list of dishes.

    Parameters:
        review (str): The review text to be processed.

    Returns:
        str: A comma-separated string of found dishes, or None if no dishes are found.
    """
    found_dishes = [dish for dish in dishes if dish.lower() in review]
    return ', '.join(found_dishes) if found_dishes else None

def get_cuisine_subgraph(cuisine, dishes_data, top_n = 10):
    """
    Generates a subgraph of the top dishes for a specific cuisine based on dish co-occurrence.

    Parameters:
        cuisine (str): The name of the cuisine to filter data by.
        dishes_data (pd.DataFrame): The dataset containing dish and cuisine information.
        top_n (int, optional): The number of top dishes to include in the subgraph. Default is 10.

    Returns:
        networkx.Graph: A subgraph containing the top dishes and their co-occurrence relationships for the specified cuisine.
    """
    cuisine_data = dishes_data[dishes_data['Cuisines'].str.contains(cuisine, na = False, case = False)]
    
    cuisine_data['Dishes_List'] = cuisine_data['Dishes'].dropna().apply(lambda x: [dish.strip() for dish in x.split(',')])
    pairs = []
    for dishes in cuisine_data['Dishes_List'].dropna():
        pairs.extend(combinations(sorted(dishes), 2))
    
    pair_counts = Counter(pairs)
    
    G = nx.Graph()
    for (dish1, dish2), weight in pair_counts.items():
        if weight > 0:  
            G.add_edge(dish1, dish2, weight=weight)
    
    dish_degrees = G.degree(weight = 'weight')
    top_dishes = sorted(dish_degrees, key=lambda x: x[1], reverse = True)[:top_n]
    top_dishes = [dish for dish, _ in top_dishes]
    return G.subgraph(top_dishes)