import os
import pickle
import random
import gensim
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from keras import Sequential
from keras.initializers import Constant
from keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, GRU, LSTM
from keras_preprocessing.text import Tokenizer
from matplotlib.ticker import MaxNLocator
from scipy.stats import mode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
#import seaborn as sn
import numpy as np
import nltk
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
from nltk.parse.corenlp import CoreNLPDependencyParser
import enchant

# name of the question text column
QUESTION_COL_NAME = 'Question'

# name of the question type column
TYPE_COL_NAME = 'Type'

# Filter out the special characters
FILTERS = '!"#$%&*+/:;<=>@[\\]^_`{|}~\t\n'

ENGLISH_DICT = enchant.Dict("en_US")

BIO_WORDS_PATH = "./data/biomed_terms.txt"

STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def bar_plot(x, heights, title, x_label, y_label, save_path, y_scale_override=None, use_int_x_ticks=False):

    """Plot and save a bar plot"""

    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=19)
    plt.bar(x, heights)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=17)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    if y_scale_override:
        plt.yscale(y_scale_override)
    if use_int_x_ticks:
        plt.xticks(range(min(x), max(x) + 1))
    plt.show(block=False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.pause(0.05)


def auto_label(rects, ax, proportions):

    """Attach a text label above each bar in *rects*, displaying its height, adapted from a Matplotlib example: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html"""

    for rect, proportion in zip(rects, proportions):
        height = rect.get_height()
        ax.annotate('{}%'.format(int(round(proportion * 100))),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


def grouped_bar_plot(data, title, y_label, save_path, types_to_ignore=None, title_size=21):

    """Plot and save a grouped bar plot, adapted from a Matplotlib example: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html"""

    labels = [datum[0] for datum in data]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots()
    rect_groups = list()
    types = [_type for _type in data[0][1].keys() if not types_to_ignore or _type not in types_to_ignore]
    for i, _type in enumerate(types):
        rect_groups.append(ax.bar(x + i * width, [datum[1][_type] for datum in data], width, label=_type))

    ax.set_title(title, fontsize=title_size)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=20)
    ax.legend(fontsize=18)
    #ax.legend(fontsize=18, loc="upper left")
    plt.tight_layout()
    plt.gcf().set_size_inches(18, 5)

    for i, (rects, type_) in enumerate(zip(rect_groups, types)):
        auto_label(rects, ax, [datum[1][type_] / (datum[1]["total"]) for datum in data])

    fig.tight_layout()

    plt.show(block=False)
    plt.pause(0.05)
    plt.savefig(save_path, bbox_inches='tight')


def check_imbalance(data_frame):

    """Generate a bar plot showcasing the number of examples present per class"""

    class_counts = data_frame.groupby(TYPE_COL_NAME)[QUESTION_COL_NAME].count()
    # print("Instances per class:\n", class_counts)
    bar_plot([("Yes/No" if string == "yesno" else string.capitalize()) for string in class_counts.index.values.tolist()], class_counts.tolist(), "Class distribution of BioASQ questions", "Number of instances", "Question Type", "class_dist.png")


def analyze_token_distributions(data_frame, type_dictionary, is_for_train=False):

    """Analyze token/sentence-related distributions"""

    # find the distribution of the number of sentences among questions
    sentence_num_occurrences = Counter([len(nltk.sent_tokenize(question)) for question in data_frame[QUESTION_COL_NAME]])

    bar_plot(sentence_num_occurrences.keys(), [sentence_num_occurrences[num] for num in sentence_num_occurrences.keys()], ("Training Set - " if is_for_train else "") + "Questions by sentence number (logarithmic scale)", "Sentence number", "Instance number", ("training_" if is_for_train else "") + "sentence_num_dist.png", "log", True)

    # word-tokenize the sentences
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in data_frame[QUESTION_COL_NAME]]

    # find the distribution of the number of tokens among questions
    token_num_occurrences = Counter([len(sentence) for sentence in tokenized_sentences])

    token_num_range = list(range(min(token_num_occurrences.keys()), max(token_num_occurrences.keys()) + 1))
    bar_plot(token_num_range, [(token_num_occurrences[num] if num in token_num_occurrences else 0) for num in token_num_range], ("Training Set - " if is_for_train else "") + "Questions by token number", "Token number", "Instance number", ("training_" if is_for_train else "") + "token_num_dist.png")

    # for each word, find the number of times it appears in each class (and in total)
    tokens_class_dict = dict()
    unique_question_types = list(type_dictionary.keys())
    unique_question_types.append("total")
    for tokens, question_type in zip(tokenized_sentences, data_frame[TYPE_COL_NAME].tolist()):
        for token in tokens:
            if token not in tokens_class_dict:
                tokens_class_dict[token] = {unique_type: 0 for unique_type in unique_question_types}
            tokens_class_dict[token]["total"] += 1
            tokens_class_dict[token][question_type] += 1
    tokens_by_occurrences = sorted(tokens_class_dict.items(), key=lambda k: k[1]["total"], reverse=True)
    print(tokens_by_occurrences)
    grouped_bar_plot(tokens_by_occurrences[:10], ("Training Set - " if is_for_train else "") + "Class distribution of the most frequent tokens (1st to 10th)", "Number of occurrences", ("training_" if is_for_train else "") + "tokens_by_class_1.png", ["total"], 18 if is_for_train else 21)
    grouped_bar_plot(tokens_by_occurrences[10:20], ("Training Set - " if is_for_train else "") + "Class distribution of the most frequent tokens (11th to 20th)", "Number of occurrences", ("training_" if is_for_train else "") + "tokens_by_class_2.png", ["total"], 18 if is_for_train else 21)
    grouped_bar_plot(tokens_by_occurrences[20:30], ("Training - " if is_for_train else "") + "Class distribution of the most frequent tokens (21th to 30th)", "Number of occurrences", ("training_" if is_for_train else "") + "tokens_by_class_3.png", ["total"], 18 if is_for_train else 21)


def analyze_data(data_frame, labels, test_proportion, type_dictionary, show_plots):

    """Perform linguistic data analysis"""

    if show_plots:

        # check if the classes are imbalanced
        check_imbalance(data_frame)

        # analyze token/sentence-related distributions
        analyze_token_distributions(data_frame, type_dictionary)

    # analyze token/sentence-related distributions for training data only
    x_train, x_test, y_train, y_test, _, _ = train_test_split(data_frame, labels, data_frame.index, test_size=test_proportion, random_state=0)

    if show_plots:

        analyze_token_distributions(x_train.assign(Type=data_frame[TYPE_COL_NAME]), type_dictionary, True)

    return x_train, x_test, y_train, y_test


def calculate_tfidf(data_frame, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english'):

    """Calculate the TF-IDF features and labels"""

    # default parameters and general usage based on: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    tfi_df = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm=norm, encoding=encoding, ngram_range=ngram_range, stop_words=stop_words)
    features = tfi_df.fit_transform(data_frame[QUESTION_COL_NAME]).toarray()
    labels = data_frame.Type_id
    return features, labels


def create_test_rule_based_model(x_test, y_test, type_dictionary):

    """Create and test the rule-based model"""

    # word-tokenize the test sentences
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in x_test[QUESTION_COL_NAME]]

    # get the true test labels
    inverted_type_dictionary = {value: key for key, value in type_dictionary.items()}
    test_labels = [inverted_type_dictionary[y] for y in y_test]

    # rules in order of confidence
    rules = [("List", "list"), ("Are", "yesno"), ("Is", "yesno"), (".", "list"), ("role", "summary"), ("a", "yesno"), ("What", "summary")]
    default_type = "factoid"

    # apply the rules to predict
    predictions = list()
    for sentence in tokenized_sentences:
        done = False
        for (token, type_) in rules:
            if token in sentence:
                predictions.append(type_)
                done = True
                break
        if not done:
            predictions.append(default_type)

    accuracy = sum([(1 if pred == real else 0) for pred, real in zip(predictions, test_labels)]) / len(test_labels) * 100
    print("Rule-based model:\nTest accuracy: {}%\n{}".format(round(accuracy, 2), metrics.classification_report(test_labels, predictions)))


def set_value(row_number, assigned_value):

    """For each row number assign the given value"""
    return assigned_value[row_number]


def compare_models(features, labels):

    """Perform the classification with three models to compare"""

    # model comparison code based on: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    models = [svm.LinearSVC(), MultinomialNB()]
    cross_val_fold = 10
    cv_df = pd.DataFrame(index=range(cross_val_fold * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=cross_val_fold)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['Model', 'fold_idx', 'Accuracy'])
    print(cv_df.groupby('Model')['Accuracy'].mean())
    plt.figure()
    plt.show(block=False)
    plt.pause(0.05)


def save_model(model, filename):

    """Save the word2vec model"""
    model.wv.save_word2vec_format(filename, binary=False)
    return filename


def build_model_fixed(embedding_dim, vocab_size, words_per_text, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix):

    """Build and compile a neural network model"""

    # use a linear layer stack
    model = Sequential()

    # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # use convolution in the text identifiers to learn association of words that appear together in the text
    model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

    # reduce dimensionality with a max-pooling layer
    model.add(GlobalMaxPooling1D())

    # increase the model complexity
    model.add(Dense(hidden_units, activation="relu"))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # the output layer has a single unit, and uses sigmoid activation function, suitable for binary classification
    model.add(Dense(4, activation="softmax"))

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


def build_complex_model_fixed(embedding_dim, vocab_size, words_per_text, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix):

    """Build and compile a neural network model"""

    # use a linear layer stack
    model = Sequential()

    # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # use convolution in the text identifiers to learn association of words that appear together in the text
    model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

    # reduce dimensionality with a max-pooling layer
    model.add(GlobalMaxPooling1D())

    # increase the model complexity
    model.add(Dense(hidden_units, activation="relu"))

    # use dropout to ignore some input units during training and reduce overfitting
    model.add(Dropout(dropout_ratio))

    # increase the model complexity
    model.add(Dense(hidden_units, activation="relu"))

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


def build_model(embedding_dim, vocab_size, words_per_text, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix, model_index):

    """Build and compile a neural network model"""

    # use a linear layer stack
    model = Sequential()

    if model_index == 0:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))


    if model_index == 1:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=filter_num*2, kernel_size=mask_size//2, activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    if model_index == 2:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=int(filter_num*1.25), kernel_size=int(mask_size/1.25), activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units//2, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    if model_index == 3:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # gated recurrent units
        model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    if model_index == 4:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # gated recurrent units
        model.add(LSTM(units=64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=filter_num, kernel_size=mask_size, activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    if model_index == 5:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # gated recurrent units
        model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))

    if model_index == 6:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio*2))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=int(filter_num*5), kernel_size=int(mask_size), activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units*2, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    if model_index == 7:

        # set up a layer of word embeddings created by mapping the word identifiers fo the texts to a high dimensional space
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=words_per_text, embeddings_initializer=Constant(embedding_matrix)))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

        # use convolution in the text identifiers to learn association of words that appear together in the text
        model.add(Conv1D(filters=int(filter_num*2), kernel_size=int(mask_size*3), activation="relu"))

        # reduce dimensionality with a max-pooling layer
        model.add(GlobalMaxPooling1D())

        # increase the model complexity
        model.add(Dense(hidden_units, activation="relu"))

        # use dropout to ignore some input units during training and reduce overfitting
        model.add(Dropout(dropout_ratio))

    # the output layer has a single unit, and uses sigmoid activation function, suitable for binary classification
    model.add(Dense(4, activation="softmax"))

    # compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


def create_word2vec(all_questions, embeddings_dim, window, workers, min_count):

    """ Create the word2vec model"""

    model = gensim.models.Word2Vec(sentences=all_questions, size=embeddings_dim, window=window, workers= workers, min_count=min_count)
    # vocab size
    words = list(model.wv.vocab)
    print("vocabulary size {}".format(len(words)))
    return model


def load_precomputed_model(filename):

    """Load a precomputed model of embeddings from a file"""

    embeddings_index = {}
    f = open(filename, encoding="utf-8")
    for i, line in enumerate(f):
        if i == 0:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def use_glove_embeddings(num_words, embedding_dim, word_index):

    # load the matrix if available
    glove_matrix_path = "./data/glove_embedding_matrix.pickle"
    if os.path.isfile(glove_matrix_path):

        with open(glove_matrix_path, "rb") as file:
            embedding_matrix = pickle.load(file)

    # otherwise build it
    else:
        #  extract word embeddings from glove
        embeddings_index = dict()
        f = open('./data/glove.twitter.27B.200d.txt')
        # line actions partially based on https://github.com/keras-team/keras/issues/6307
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # partially based on https://stackoverflow.com/q/56880252 (same for the rest of embedding matrix loadings)
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, index in word_index.items():
            if index > num_words - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

        # save matrix for later use
        with open(glove_matrix_path, "wb") as file:
            pickle.dump(embedding_matrix, file)

    return embedding_matrix


def use_biowordvec_embeddings(num_words, embedding_dim, word_index, sentences):

    # load the matrix if available
    bio_matrix_path = "./data/bio_embedding_matrix.pickle"
    if os.path.isfile(bio_matrix_path):

        with open(bio_matrix_path, "rb") as file:
            embedding_matrix = pickle.load(file)

    # otherwise build it
    else:

        # load the model
        embeddings_path = "./data/bio_embedding_extrinsic"
        vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

        # fine-tune, as done on: https://datascience.stackexchange.com/a/32433
        model = Word2Vec(size=embedding_dim, min_count=1)
        model.build_vocab(sentences=sentences)
        model.build_vocab([list(vectors.vocab.keys())], update=True)
        model.intersect_word2vec_format(embeddings_path, binary=True, lockf=1.0)
        model.train(sentences, total_examples=len(sentences), epochs=10)

        out_of_vocab_num = 0

        bio_words_set = set()

        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, index in word_index.items():
            if index > num_words - 1:
                break
            else:
                if word in model:
                    embedding_vector = model[word]
                    bio_words_set.add(word)
                else:
                    embedding_vector = np.zeros(embedding_dim)
                    out_of_vocab_num += 1
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

        out_of_vocab_proportion = out_of_vocab_num / num_words
        print("Percentage of out-of-vocabulary words in pre-trained embeddings: {}%".format(round(out_of_vocab_proportion * 100, 2)))

        # save matrix for later use
        with open(bio_matrix_path, "wb") as file:
            pickle.dump(embedding_matrix, file)

        # save biomedical words set for later use
        with open(BIO_WORDS_PATH, "wb") as file:
            pickle.dump(bio_words_set, file)

    return embedding_matrix


def lower(tokens):

    """Lower-case all tokens of a list"""

    return [token.lower() for token in tokens]


def lower_but_first(tokens):

    """Lower-case all tokens of a list except for first one"""

    return [(tokens[i] if i == 0 else tokens[i].lower()) for i in range(len(tokens))]


def embedding_to_matrix(data_frame, question_sentences, embedding_index, embedding_dim, max_length, min_num_words=0, test_index_range=None, valid_index_range=None):

    """Generate an embedding matrix from word embeddings"""
    test_split = 0.1

    valid_data = None

    use_lower_and_upper = False

    # use NLTK tokenizer, lower-casing every word but the first of each sentence
    if use_lower_and_upper:
        tokenized_question_sentences = [lower_but_first(nltk.word_tokenize(sent)) for sent in question_sentences]
    else:
        #tokenized_question_sentences = [nltk.pos_tag(lower(nltk.word_tokenize(sent))) for sent in question_sentences]
        tokenized_question_sentences = [nltk.word_tokenize(sent) for sent in question_sentences]

    question_postags = [nltk.pos_tag(lower(sent)) for sent in tokenized_question_sentences]
    question_postags = [[pos[1] for pos in tags] for tags in question_postags]
    print(question_postags[:10])

    triple_file_name = "./data/corenlp_chunks.pkl"
    if os.path.isfile(triple_file_name):

        with open(triple_file_name, "rb") as file:
            question_triples = pickle.load(file)
    else:
        question_triples = list()
        parser = CoreNLPDependencyParser()
        for question in question_sentences:
            tree = next(parser.raw_parse(question))
            question_triples.append([triple[0][1] + "-" + triple[1] + "-" + triple[2][1] for triple in tree.triples()])
        print(question_triples)
        print(len(question_triples))
        with open(triple_file_name, "wb") as file:
            pickle.dump(question_triples, file)

    # Vectorize the text samples into a 2D integer tensor
    tokenizer_obj = Tokenizer(lower=not use_lower_and_upper, filters=FILTERS)
    tokenizer_obj.fit_on_texts(tokenized_question_sentences)
    sequences = tokenizer_obj.texts_to_sequences(tokenized_question_sentences)

    # vectorize postags
    pos_tokenizer_obj = Tokenizer(lower=False, filters=FILTERS)
    pos_tokenizer_obj.fit_on_texts(question_postags)
    pos_sequences = pos_tokenizer_obj.texts_to_sequences(question_postags)

    # vectorize triples -- pos-chunk-pos
    triple_tokenizer_obj = Tokenizer(lower=False, filters=FILTERS)
    triple_tokenizer_obj.fit_on_texts(question_triples)
    triple_sequences = triple_tokenizer_obj.texts_to_sequences(question_triples)

    # pad sequences
    word_index = tokenizer_obj.word_index
    print('Found %s unique tokens.' % len(word_index))
    question_pad = pad_sequences(sequences, maxlen=max_length)
    question_type = data_frame['Type_id'].values

    # perform above operation for pos tag
    pos_index = pos_tokenizer_obj.word_index
    pos_pad = pad_sequences(pos_sequences, maxlen=max_length)

    # perform above operation for triples
    triple_index = triple_tokenizer_obj.word_index
    triple_pad = pad_sequences(triple_sequences, maxlen=max_length)

    use_glove = False
    use_biowordvec = False
    combine_embeddings = False
    use_pos = False
    combine_pos = False
    use_triples = False
    combine_triples = False
    combine_pos_triples = False

    pad = question_pad
    if use_pos:
        pad = pos_pad
    elif use_triples:
        pad = triple_pad

    # use the last indices for testing if a range is not specified, based on a split size
    if not test_index_range:

        # split the data into a training set and a test set
        indices = np.arange(pad.shape[0])
        #random.Random(143).shuffle(indices)
        random.Random(486).shuffle(indices)
        #np.random.huffle(indices)
        pad = pad[indices]
        question_type = question_type[indices]

        test_sample_num = int(test_split * pad.shape[0])
        x_train_pad = pad[:-test_sample_num]
        y_train = question_type[:-test_sample_num]
        x_test_pad = pad[-test_sample_num:]
        y_test = question_type[-test_sample_num:]

    # otherwise use the specified index range for testing
    else:
        train_end_index = valid_index_range[0] if valid_index_range else test_index_range[0]
        x_train_pad = np.concatenate((pad[:train_end_index], pad[test_index_range[1]:]), axis=0)
        y_train = np.concatenate((question_type[:train_end_index], question_type[test_index_range[1]:]), axis=0)
        if valid_index_range:
            x_valid_pad = pad[valid_index_range[0]:valid_index_range[1]]
            y_valid = question_type[valid_index_range[0]:valid_index_range[1]]
            valid_data = [x_valid_pad, y_valid]
        x_test_pad = pad[test_index_range[0]:test_index_range[1]]
        y_test = question_type[test_index_range[0]:test_index_range[1]]

    num_words = max(len(word_index) + 1, min_num_words)

    embedding_matrix = None
    wordvec_matrix = None
    glove_matrix = None
    biowordvec_matrix = None
    pos_matrix = None

    if use_glove or combine_embeddings:
        print("Using glove embeddings")
        final_embedding_dim = 200
        glove_matrix = use_glove_embeddings(num_words, final_embedding_dim, word_index)
        if not combine_embeddings:
            embedding_matrix = glove_matrix

    if use_biowordvec or combine_embeddings or combine_pos or combine_triples or combine_pos_triples:
        print("Using BioWordVec")
        final_embedding_dim = 200
        biowordvec_matrix = use_biowordvec_embeddings(num_words, final_embedding_dim, word_index, question_sentences)
        if not combine_embeddings:
            embedding_matrix = biowordvec_matrix

    if (not use_glove and not use_biowordvec) or combine_embeddings:

        '''if combine_embeddings:
            embedding_dim = 200'''
        wordvec_matrix = np.zeros((num_words, embedding_dim))

        for word, i in word_index.items():
            if i > num_words:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                wordvec_matrix[i] = embedding_vector

        if not combine_embeddings:
            embedding_matrix = wordvec_matrix

    if combine_embeddings:

        print(wordvec_matrix.shape)
        print(glove_matrix.shape)
        print(biowordvec_matrix.shape)
        embedding_matrix = np.concatenate((glove_matrix, biowordvec_matrix, wordvec_matrix), axis=1)

    if use_pos or combine_pos or combine_pos_triples:

        pos_model = Word2Vec(question_postags, size=embedding_dim)

        pos_matrix = np.zeros((num_words, embedding_dim))

        for pos, i in pos_index.items():
            if i > num_words:
                continue
            if pos in pos_model:
                embedding_vector = pos_model[pos]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    pos_matrix[i] = embedding_vector

        if combine_pos:
            embedding_matrix = np.concatenate((biowordvec_matrix, pos_matrix), axis=1)

        else:
            embedding_matrix = pos_matrix

    if use_triples or combine_triples or combine_pos_triples:

        triple_model = Word2Vec(question_triples, size=embedding_dim)

        triple_matrix = np.zeros((num_words, embedding_dim))

        for triple, i in triple_index.items():
            if i > num_words:
                continue
            if triple in triple_model:
                embedding_vector = triple_model[triple]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    triple_matrix[i] = embedding_vector

            embedding_matrix = triple_matrix

        if combine_triples:
            embedding_matrix = np.concatenate((biowordvec_matrix, triple_matrix), axis=1)

        elif combine_pos_triples:
            embedding_matrix = np.concatenate((biowordvec_matrix, pos_matrix, triple_matrix), axis=1)

    final_embedding_dim = embedding_matrix.shape[1]

    return x_train_pad, y_train, x_test_pad, y_test, embedding_matrix, num_words, final_embedding_dim, valid_data


def create_train_nn(x_train, y_train, x_test, y_test, num_words, embeddings_dim, embedding_matrix, max_length, type_dictionary, use_complex_model=False, valid_data=None):

    """Create and train a neural network model"""
    # model parameters
    filter_num = 300
    mask_size = 5
    dropout_ratio = 0.15
    hidden_units = 500
    optimizer = "nadam"
    loss_function = "sparse_categorical_crossentropy"

    use_ensemble = False
    discard_recurrent_networks = False
    if use_ensemble:

        y_predictions = [list() for _ in range(len(y_test))]

        # create an ensemble with different models
        model_indices = [0, 1, 2, 6, 7] if discard_recurrent_networks else range(8)
        for model_index in model_indices:

            # build a specific model
            model = build_model(embeddings_dim, num_words, max_length, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix, model_index)

            # training parameters
            batch_size = 25
            epoch_num = 7

            # show the training parameters
            print("Training parameters:\n\tBatch size:\t{}\n\tEpochs:\t\t{}\n".format(batch_size, epoch_num))

            # define the size of the validation set
            valid_proportion = 0.04
            valid_size = round(valid_proportion * len(x_train))

            # extract training from validation and change notation
            x_train = np.array(x_train[valid_size:])
            y_train = y_train[valid_size:]
            x_valid = np.array(x_train[:valid_size])
            y_valid = y_train[:valid_size]
            x_test = np.array(x_test)
            y_test = y_test

            # train the model
            model.fit(x_train, y_train, validation_data=[x_valid, y_valid], batch_size=batch_size, epochs=epoch_num)

            # keep the predictions of the model
            y_prediction = model.predict_classes(x_test)
            for i, y in enumerate(y_prediction):
                y_predictions[i].append(y)

        # perform majority vote among the predictions of the ensemble models
        y_prediction = [mode(pred)[0][0] for pred in y_predictions]

        correct_prediction_num = 0
        for pred, actual in zip(y_prediction, y_test):
            if pred == actual:
                correct_prediction_num += 1
        accuracy = correct_prediction_num / len(y_prediction)
        print("Deep Learning Ensemble test accuracy: ", round(accuracy * 100, 3))

    else:

        # build a CNN model
        if False:#use_complex_model:
            model = build_complex_model_fixed(embeddings_dim, num_words, max_length, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix)
        else:
            model = build_model_fixed(embeddings_dim, num_words, max_length, filter_num, mask_size, dropout_ratio, hidden_units, optimizer, loss_function, embedding_matrix)

        print('Summary of the built model...')
        print(model.summary())

        print('Train...')

        epoch_num = 15 #15  # 15  #20 #7 #10
        batch_size = 256
        history = model.fit(x_train, y_train, validation_data=valid_data, batch_size=batch_size, epochs=epoch_num, verbose=1)
        # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, validation_data=(x_test, y_test), verbose=1)
        print('Testing...')
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

        print('Test score:', score)
        print('Test accuracy:', acc)

        print("Accuracy: {0:.2%}".format(acc))

        scores = model.predict(x_test, batch_size=batch_size)
        predicted_y = [list(score).index(max(score)) for score in scores]
        print(predicted_y)
        print("Neural model results:\n", metrics.classification_report(y_test, predicted_y))

        return model, history.history["val_accuracy"][-1] if valid_data else None


def create_train_test_machine_learning(data_frame, test_proportion, features, labels):

    """Create, train and test Machine Learning methods"""

    x_train, x_test, y_train, y_test, _, _ = train_test_split(features, labels, data_frame.index, test_size=test_proportion, random_state=0)

    # apply linear Support Vector Classifier with the TF-IDF features
    model = svm.LinearSVC()
    model.fit(x_train, y_train)
    predicted_y = model.predict(x_test)
    final_model_accuracy = accuracy_score(y_test, predicted_y)
    print("SVM results:\n", metrics.classification_report(y_test, predicted_y))
    model_name = model.__class__.__name__
    print("Accuracy obtained by the model {} is {}".format(model_name, final_model_accuracy))

    # compare different models
    compare_models(features, labels)

    return x_train, x_test, y_train, y_test


def create_train_test_deep_learning(data_frame, type_dictionary, min_max_length=0, min_num_words=0, test_index_range=None, use_complex_model=False, valid_index_range=None):

    """Create, train and test Deep Learning methods"""

    # get all questions
    all_questions = data_frame.loc[:, QUESTION_COL_NAME].values

    # find the maximum question length for later padding
    cal_max_length = max([len(s.split()) for s in all_questions])
    max_length = max(cal_max_length, min_max_length)
    print("The maximum length of a question is ", max_length)

    # word embedding parameters
    embeddings_dim = 128
    window = 5
    workers = 4
    min_count = 1

    # loading/creation and saving of the word2vec model
    embeddings_file_name = "./data/embeddings_word2vec.txt"
    if not os.path.isfile(embeddings_file_name):
        model = create_word2vec(all_questions, embeddings_dim, window, workers, min_count)
        save_model(model, embeddings_file_name)

    # load precomputed word embeddings into dictionary
    embeddings_index = load_precomputed_model(embeddings_file_name)

    # generate the embedding matrix from word embeddings for training and test
    x_train, y_train, x_test, y_test, embedding_matrix, num_words, embeddings_dim_override, valid_data = embedding_to_matrix(data_frame, all_questions, embeddings_index, embeddings_dim, max_length, min_num_words, test_index_range, valid_index_range)

    if embeddings_dim_override > 0:
        embeddings_dim = embeddings_dim_override

    # create and train the network
    model, valid_acc = create_train_nn(x_train, y_train, x_test, y_test, num_words, embeddings_dim, embedding_matrix, max_length, type_dictionary, use_complex_model, valid_data)

    return model, max_length, valid_acc


def is_text_biomedical(text, bio_words_set):

    if type(text) != str:

        return False

    biomed_token_num = 0

    tokens = nltk.word_tokenize(text)

    for token in tokens:

        # check if token is biomedical
        if token not in STOPWORDS and token.lower() in bio_words_set:

            biomed_token_num += 1

    threshold = 0.38
    return biomed_token_num >= len(tokens) * threshold


def filter_non_bio_texts(quora_data_frame):

    bio_quora_file_path = "./data/quora_bio_questions.tsv"
    if os.path.isfile(bio_quora_file_path):
        quora_data_frame = pd.read_csv(bio_quora_file_path, sep='\t')

    else:

        bio_words_set = set()
        with open(BIO_WORDS_PATH, "r") as file:
            words = file.readlines()
            for word in words:
                bio_words_set.add(word.strip().lower())
            #print(bio_words_set)
            print("Num of biomedical terms:", len(words))
        quora_data_frame = quora_data_frame[quora_data_frame["question1"].apply(lambda text: is_text_biomedical(text, bio_words_set))]
        quora_data_frame = quora_data_frame[quora_data_frame["question2"].apply(lambda text: is_text_biomedical(text, bio_words_set))]
        quora_data_frame.to_csv(bio_quora_file_path, sep="\t")
        print("biomedical-filtered quora data set length", len(quora_data_frame))

    return quora_data_frame


def get_predictions_from_model(model, data_frame, max_length):
    """ Get the predictions of the questions with passed model"""
    y_predictions = list()
    for question_number in ["question1", "question2"]:
        # preprocess the data
        tokenized_questions = [(list() if type(sent) != str else nltk.word_tokenize(sent)) for sent in data_frame[question_number].tolist()]

        # Vectorize the text samples into a 2D integer tensor
        tokenizer_obj = Tokenizer(lower=True, filters=FILTERS)
        tokenizer_obj.fit_on_texts(tokenized_questions)
        sequences = tokenizer_obj.texts_to_sequences(tokenized_questions)

        x_text = pad_sequences(sequences, maxlen=max_length)

        scores = model.predict(x_text, batch_size=256)
        y_predictions.append([list(score).index(max(score)) for score in scores])
    return y_predictions


def get_prediction_matches(model, data_frame, max_length):
    preds = get_predictions_from_model(model, data_frame, max_length)

    matches = sum(a == b for a,b in zip(preds[0], preds[1]))
    match_percent = matches/len(preds[0])*100
    print("percentage of correct matches ", match_percent)

    return matches, match_percent


def main():

    """Main function"""

    task1 = False

    file_path = "./data/Questions.xlsx"
    data_frame = pd.read_excel(file_path)

    # convert the categorical classes to numeric classes
    type_dictionary = {'yesno': 0, 'summary': 1, 'list': 2, 'factoid': 3}
    data_frame['Type_id'] = data_frame[TYPE_COL_NAME].apply(set_value, args=(type_dictionary, ))

    # generate the tf idf features for the questions
    features, labels = calculate_tfidf(data_frame)
    # print(features.shape)

    test_proportion = 0.1

    if task1:

        show_plots = True

        # analyze the data
        x_train, x_test, y_train, y_test = analyze_data(data_frame, labels, test_proportion, type_dictionary, show_plots)

        # create and test the rule-based model
        create_test_rule_based_model(x_test, y_test, type_dictionary)


        # create, train and test Machine Learning methods
        create_train_test_machine_learning(data_frame, test_proportion, features, labels)

        # create, train and test Deep Learning methods
        model = create_train_test_deep_learning(data_frame, type_dictionary)

    else:

        # perform task 2
        min_num_words = 30000

        quora_processed_data = "./data/quora_processed_data.tsv"
        if False: #os.path.isfile(quora_processed_data):
            # load the processed quora data
            quora_data_frame = pd.read_csv(quora_processed_data, sep='\t')
            print(len(quora_data_frame))

            max_lengths = list()
            max_lengths.append(max([len(s.split()) for s in quora_data_frame["question1"].tolist()]))
            max_lengths.append(max([len(s.split()) for s in quora_data_frame["question2"].tolist()]))
            max_length = max(max_lengths)

            # start the co-training with 1000 instances per step
            i = 0
            while i < len(quora_data_frame):
                # create, train and test Deep Learning methods

                model, max_length = create_train_test_deep_learning(data_frame, type_dictionary, max_length, min_num_words)

        else:
            # read the original quora data and process it
            # read the data frame
            quora_file_path = "./data/quora_duplicate_questions.tsv"
            quora_data_frame = pd.read_csv(quora_file_path, sep='\t')
            print("original quora data set length", len(quora_data_frame))

            # filter out non-biomedical texts
            quora_data_frame = filter_non_bio_texts(quora_data_frame)

            # Get all the duplicate texts
            duplicate_data_frame = quora_data_frame[quora_data_frame['is_duplicate'] == 1]
            print(len(duplicate_data_frame))
            print(duplicate_data_frame.columns)
            # print("quora data set length after removing the duplicate question rows", len(quora_data_frame))

            test_split = 0.1
            valid_split = 0.1

            indices = np.arange(len(data_frame))
            random.Random(486).shuffle(indices)
            data_frame = data_frame.reindex(indices)

            test_sample_num = int(test_split * len(data_frame))
            test_index_range = (len(data_frame) - test_sample_num, len(data_frame))

            valid_sample_num = int(valid_split * len(data_frame))
            valid_index_range = (len(data_frame) - test_sample_num - valid_sample_num, len(data_frame) - - test_sample_num )

            max_lengths = list()
            max_lengths.append(max([len(s.split()) for s in duplicate_data_frame["question1"].tolist()]))
            max_lengths.append(max([len(s.split()) for s in duplicate_data_frame["question2"].tolist()]))
            max_length = max(max_lengths)

            # create, train and test Deep Learning methods
            model, max_length, _ = create_train_test_deep_learning(data_frame, type_dictionary, max_length, min_num_words, test_index_range=test_index_range)

            # predict the quora data set questions
            quora_preds = get_predictions_from_model(model, quora_data_frame, max_length)
            quora_data_frame["prediction_match"] = [i if i == j else False for i, j in zip(quora_preds[0], quora_preds[1])]
            #quora_data_frame["prediction_match"] = [i == j for i, j in zip(quora_preds[0], quora_preds[1])]
            quora_data_frame = quora_data_frame[quora_data_frame["prediction_match"] != False]
            quora_usable_num = len(quora_data_frame)
            print("length after deleting", quora_usable_num)

            class_counts = quora_data_frame.groupby("prediction_match")["question1"].count()
            inverted_type_dictionary = {value: key for key, value in type_dictionary.items()}
            bar_plot([inverted_type_dictionary[question_type].capitalize() for question_type in class_counts.index.values.tolist()], class_counts.tolist(), "Class distribution of Quora biomedical questions", "Number of instances", "Question Type", "quora_class_dist.png")

            # get the predictions of the duplicate data frame
            matches, match_percent = get_prediction_matches(model, duplicate_data_frame, max_length)
            print("Correct matches percentage for duplicated questions: {}%".format(match_percent))


            # perform co-training
            initial_val = 0
            inc = quora_usable_num // 10
            inc_num = 0

            true_accs = []
            accepted_accs = []

            valid_acc = 0
            while initial_val < len(quora_data_frame):
                inc_num += 1
                old_valid_acc = valid_acc
                model, max_length, valid_acc = create_train_test_deep_learning(data_frame, type_dictionary, max_length, min_num_words, test_index_range, True, valid_index_range)
                quora_data_frame_inc = quora_data_frame[initial_val:initial_val+inc]
                '''quora_preds = get_predictions_from_model(model, quora_data_frame_inc, max_length)
                quora_data_frame_inc["prediction_match"] = [i if i == j else False for i, j in zip(quora_preds[0], quora_preds[1])]
                quora_data_frame_inc = quora_data_frame_inc[quora_data_frame_inc["prediction_match"]!=False]'''
                initial_val += inc
                for i,j in zip(quora_data_frame_inc["question1"], quora_data_frame_inc["prediction_match"]):
                    data_frame = data_frame.append({'Question':i, 'Type_id':j}, ignore_index=True)
                for i,j in zip(quora_data_frame_inc["question2"], quora_data_frame_inc["prediction_match"]):
                    data_frame = data_frame.append({'Question':i, 'Type_id':j}, ignore_index=True)
                print("Valid acc: ", valid_acc)
                print("Has increased valid acc?", valid_acc > old_valid_acc)
                true_accs.append(valid_acc)
                if valid_acc <= old_valid_acc:
                    valid_acc = old_valid_acc
                    for i,j in zip(quora_data_frame_inc["question1"],quora_preds[0]):
                        data_frame = data_frame[data_frame["Question"] != i]
                    for i,j in zip(quora_data_frame_inc["question2"],quora_preds[1]):
                        data_frame = data_frame[data_frame["Question"] != i]
                accepted_accs.append(valid_acc)

                # get the predictions of the duplicate data frame
                matches, match_percent = get_prediction_matches(model, duplicate_data_frame, max_length)
                print("Correct matches percentage for duplicated questions: {}%".format(match_percent))

            # plot the validation accuracy
            plt.figure(figsize=(10, 10))
            x = np.arange(inc_num)
            plt.plot(x, accepted_accs, color="green", marker="o")
            plt.plot(x, true_accs, linestyle="--", color="orange", marker="o")
            plt.title("Validation accuracy obtained during co-training for BioWordVec model", fontsize=20)
            plt.xticks(x, x)
            plt.gca().tick_params(axis='both', which='major', labelsize=15)
            plt.xlabel("Increment step", fontsize=19)
            plt.ylabel("Validation accuracy", fontsize=19)
            plt.legend(['Preserved validation accuracy', 'Candidate validation accuracy'], loc='lower left', fontsize=18)
            plt.show(block=False)
            plt.savefig("task2  _acc_word2vec.png", bbox_inches='tight')
            plt.pause(0.05)


if __name__ == '__main__':
    main()
