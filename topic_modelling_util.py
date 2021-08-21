# Based on Computational social media - Assignement 3 - Daniel Gattica Perez - DH-500 - Spring 2020
# and on Introduction to Machine Learning for DH - Homework 11 - Mathieu Salzmann - DH-406 - Fall 2019

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Ce code nécessite un package spécifique de spacy (`python3 -m spacy download fr_core_news_md`). Se reporter à # https://stackoverflow.com/questions/13131139/lemmatize-french-text

import pandas as pd
import numpy as np
import string
import spacy
import re
import sklearn.feature_extraction.text
import sklearn.decomposition

# Init
french_nlp = spacy.load("fr_core_news_md")

# Preprocessing
def topic_modelling_preprocessing(text):
    """
    Typical topic modelling pipeline preprocessing of a given text.
    text : the text to transform
    Return : a list of tokens
    """
    
    #Lowering
    text = text.lower()
    
    #Punctuation removal   
    text = re.sub("[{}]".format(string.punctuation)," ", text)
    text = re.sub(" ( )*"," ", text)
    
    #Converting to list of words and lemmmatize
    lemmas = [token.lemma_ for token in french_nlp(text)]
    
    #Removing stopwords
    words = [w for w in lemmas if w not in spacy.lang.fr.stop_words.STOP_WORDS]
    
    return words

# LDA data generation
def filter_eligible_words(df, lowerbound_absolute, upperbound_relative):
    """
    Keep only the words that appear enough time but not too often.
     df : a dataframe with two column [cote, tokens]
         cote : the cote of the document
         tokens : the liste of preprocessed tokens
     lowerbound_absolute : the absolute number of occurences under which the token is not kept
     upperbound_relative : the ratio <number of occurences>/<number of documents> above which the token is not kept
    """
    forbidden_words = ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre",
                       "novembre","décembre","directoire","exécutif","sénat","lucerne","mousson","ministre",
                      "accepter","résoudre","déclarer","législatif","urgence"]
    
    assert upperbound_relative <= 1
    
    print("Creating tokens list without duplicates...")
    df["tokens_no_duplicates"] = df["tokens"].apply(lambda l : list(set(l)))
    
    tokens_df = df.explode(column = "tokens_no_duplicates")
    
    tokens_count_df = tokens_df.groupby("tokens_no_duplicates").count().reset_index().rename(
        columns = {"cote":"quantity", "tokens_no_duplicates":"token"})[["token","quantity"]]
    
    upperbound_absolute = upperbound_relative*len(df)
    
    print("Filtering the tokens based on number of occurencies...")
    tokens_to_keep_df = tokens_count_df[(tokens_count_df["quantity"] >= lowerbound_absolute) &
                                       (tokens_count_df["quantity"] <= upperbound_absolute)]
    
    #Suppression des nombres
    tokens_to_keep_df = tokens_to_keep_df[tokens_to_keep_df["token"].apply(lambda t : re.search("\d",t) is None)]
    
    #Supression des mots qui brouillent les pistes
    tokens_to_keep_df = tokens_to_keep_df[tokens_to_keep_df["token"].apply(lambda t : t not in forbidden_words)]
    
    tokens_to_keep = tokens_to_keep_df["token"].tolist()
    
    print("Length of the token list : {}".format(len(tokens_to_keep)))
    
    return set(tokens_to_keep)


def generate_LDA_vector(df, lowerbound_absolute, upperbound_relative):
    """
    Generate a bag of words usable with an lda model
    df : a dataframe with two column [cote, tokens]
         cote : the cote of the document
         tokens : the liste of preprocessed tokens
     lowerbound_absolute : the absolute number of occurences under which the token is not kept
     upperbound_relative : the ratio <number of occurences>/<number of documents> above which the token is not kept
     Return (doc_matrix, tokens_index)
     doc_matrix : the M_documents X N_tokens matrix
     tokens_index : a list of all valid tokens whose position is the same as the one in the model matrix
    """
    print("Defining eligible tokens...")
    eligible_tokens = filter_eligible_words(df, lowerbound_absolute, upperbound_relative)
    
    #print("Filtering the eligible tokens...")
    #df["legitimate_tokens"] = df["tokens"].apply(lambda row_tokens : # set intersection for time complexity reason
    #[t for t in row_tokens if t in eligible_tokens.intersection(set(row_tokens))])
    
    print("Counting vectorizing...")
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(vocabulary = eligible_tokens)
    
    # Le count-vectorizer prend normalement des textes, pas des listes de token déjà préprocessés, d'où le trick
    # doc_matrix est de type M_documents X N_mots
    doc_matrix = count_vectorizer.fit_transform(df["tokens"].apply(lambda l : " ".join(l)))
    
    # La position dans la matrice de chaque token. Sera utile pour plus tard pour qualifier les topics
    tokens_index = count_vectorizer.get_feature_names()
    
    return doc_matrix, tokens_index


# LDA model and components creation
def create_lda_model_and_stuff(df, n_topics, lowerbound_absolute, upperbound_relative):
    """
    Create a lda model, the probabilities to belong to each topic and the tokens position index
    df : the dataframe containing at least two columns [cote, tokens]
    n_topics : the number of topics to extract
    lowerbound_absolute : the minimum number of document in which a given token must be found to be used
    upperbound_relative : the maximum proportion of the total number of documents in which a given token ibid
    Return lda, proba, tokens_index
    lda : the sklearn lda model
    proba : a M_documents X N_topics matrix which represent the probability to belong to each topic for each doc
    tokens_index : the list of all token. Their position in the list is their position in the model matrix
    """
    
    doc_matrix, tokens_index = generate_LDA_vector(df, lowerbound_absolute, upperbound_relative)
    
    lda = sklearn.decomposition.LatentDirichletAllocation(
    n_components=n_topics, max_iter=20, learning_method='online', random_state=0, verbose = 1)
    proba = lda.fit_transform(doc_matrix)
    
    return lda, proba, tokens_index


# LDA use and exploration
def print_top_words(model, feature_names, n_top_words):
    """
    Print the n most frequent words of each topic of a given lda model. (@author Mathieu Salzmann)
    model : the lda model
    feature_names : the tokens index that say which token is at which column of the model matrix
    n_top_words : the number of words to print for each topic
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def choose_most_probable_topics(proba_row, topic_names, n_topics_to_choose):
    
    useless_topic = "x"
    
    probability_ranked_indexes = np.flip(proba_row.argsort())    
    ordrered_topics = np.array(topic_names)[probability_ranked_indexes]    
    ordrered_valid_topics = np.array(list(filter(lambda el : el != useless_topic, ordrered_topics)))    
    chosen_topics = ordrered_valid_topics[:min(n_topics_to_choose,len(ordrered_valid_topics))]
    
    return list(chosen_topics)