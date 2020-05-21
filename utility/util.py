import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlalchemy as sq
import pickle as pkl
import gensim as gs
import sklearn.decomposition as dc
import sklearn.manifold as ma
import requests as rq
import sys as sy

def sorted_dictionary(dictionary, by_val = True, ascending = True):
    """
    Sort the given dictionary
    :param dictionary: The dictionary to sort
    :param by_val: Whether or not to sort by value (Default: True)
    :param ascending: Whether or not to sort in ascending order (Default: True)
    :return: Get the list of sorted tuples
    """

    return list(sorted(dictionary.items(), key=lambda x: x[1 if by_val else 0], reverse=not ascending))

def to_txt(text, file_name):
    """
    Writes given string to given file as ASCII
    :param text: The string
    :param file_name: The file name
    """

    with open(file_name, "w") as text_file:
        text_file.write(text)

def update_progress(current, total, bar_length = 50):
    """
    Updates the terminal with an ASCII progress bar representing the percentage of work done
    :param current: The number of elements processed
    :param total: The total number of elements
    :param bar_length: The bar length in characters (Default: 50)
    """

    num_blocks = round(current / total * bar_length)
    done = ''.join([char * num_blocks for char in '#'])
    not_done = ''.join([char * (bar_length - num_blocks) for char in ' '])
    printover(f'[{done}{not_done}] - {current}/{total}')

def printover(text):
    """
    Print over the last printed line
    :param text: The text to print
    """

    sy.stdout.write('\r' + text)

def download_gdrive_file(file_id, output_file_path):
    """
    Download a file from Google Drive given its file id
    (Source: https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python)
    :param file_id: The file id
    :param output_file_path: The path of the output file
    """

    URL = "https://docs.google.com/uc?export=download"

    session = rq.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = __get_confirm_token__(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    __save_response_content__(response, output_file_path)

def show_2d_tsne(X, Y, colors):
    """
    Perform t-SNE dimensionality reduction to 2 on given data and plot it in given colors
    :param X: The vectors to reduce the dimensionality of
    :param Y: The vector labels
    :param colors: The label colors (len(colors) should equal distinct(Y))
    """

    Y_distinct = set(Y)
    distinctnum_Y = len(Y_distinct)
    num_colors = len(colors)
    if distinctnum_Y != num_colors:
        raise ValueError(   'Number of distinct Y values (' + str(distinctnum_Y) + ') ' +
                            'must equal number of colors (' + str(num_colors) + ')')

    dim_reduced_data = ma.TSNE(n_components=2).fit_transform(X)

    finalDf= pd.DataFrame(data=dim_reduced_data, columns=['principal component 1', 'principal component 2'])
    finalDf['Category'] = pd.Series(Y)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dimension 1', fontsize=15)
    ax.set_ylabel('Dimension 2', fontsize=15)
    ax.set_title('2 Dimensional t-SNE', fontsize=20)

    color_labels = ['Is_' + str(y) for y in Y_distinct]
    for y_val, color in zip(Y_distinct, colors):
        indicesToKeep = finalDf['Category'] == y_val
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=5)
    ax.legend(color_labels)
    ax.grid()

    plt.show()

def show_2d_pca(X, Y, colors):
    """
    Reduces dimensionality to 2 and shows the result of the PCA
    :param X: The input vectors
    :param Y: The output objects
    :param colors: The list of colors for the different types of outputs (len(colors) should equal distinct(Y))
    """

    Y_distinct = set(Y)
    distinctnum_Y = len(Y_distinct)
    num_colors = len(colors)
    if distinctnum_Y != num_colors:
        raise ValueError(   'Number of distinct Y values (' + str(distinctnum_Y) + ') ' +
                            'must equal number of colors (' + str(num_colors) + ')')

    pca = dc.PCA(n_components=2)

    principalComponents = pca.fit_transform(X)
    finalDf= pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf['Category'] = pd.Series(Y)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    color_labels = ['Is_' + str(y) for y in Y_distinct]
    for y_val, color in zip(Y_distinct, colors):
        indicesToKeep = finalDf['Category'] == y_val
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=5)
    ax.legend(color_labels)
    ax.grid()

    plt.show()

def try_word2vec(word):
    """
    Gets the word vector for the given work based on Google's trained model.
    1. Tries the cache first
    2. Loads the model between 0 and 1 times per run
    3. Updates cache
    :param word: The word to vectorize
    :return: The word vector
    """

    global google_word2vec_model
    global word2vec_cache
    model_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\nl\\GoogleWord2VecModel.bin'
    cache_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\nl\\word2vec_cache.pkl'

    # Check cache
    if word2vec_cache is None:
        if os.path.exists(cache_filename):
            word2vec_cache = read_pkl(cache_filename)
        else:
            word2vec_cache = {}
            to_pkl(word2vec_cache, cache_filename)

    # Try cache
    if word in word2vec_cache:
        return word2vec_cache[word], word2vec_cache[word] is not None
    # Use Google's model
    else:
        if google_word2vec_model is None:
            print('Need to load Google word2vec Model')

            # Check if model exists, download otherwise
            if not os.path.exists(model_filename):
                print('Google word2vec model not found. Will download (~3.5GB)...')
                download_gdrive_file('1kzCpXqZ_EILFAfK4G96QZBrjtezxjMiO', model_filename) # Hard-coded file id
                print('Done downloading Google word2vec model')

            print('Loading Google word2vec model...')
            google_word2vec_model = gs.models.KeyedVectors.load_word2vec_format(model_filename, binary=True)
            print('Done loading Google word2vec model')

        try:
            word2vec_cache[word] = google_word2vec_model[word]
            to_pkl(word2vec_cache, cache_filename)
            return word2vec_cache[word], True
        except:
            word2vec_cache[word] = None
            to_pkl(word2vec_cache, cache_filename)
            return word2vec_cache[word], False

def read_pkl(file_name):
    """
    De-serializes a pickle file into an object and returns it
    :param file_name: The name of the pickle file
    :return: The object that is de-serialized
    """

    with open(file_name, 'rb') as file:
        return pkl.load(file)

def to_pkl(obj, file_name):
    """
    Save the given object as a pickle file to the given file name
    :param obj: The object to serialize
    :param file_name: The file name to save it to
    :return: returns the same object back
    """

    with open(file_name, 'wb') as file:
        pkl.dump(obj, file)

def read_db(database_filename, table_name):
    """
    Read a db file and return it as a dataframe
    :param database_filename: The DB file path
    :param table_name: The table name
    :return: The dataframe
    """

    engine = sq.create_engine('sqlite:///' + database_filename)
    return pd.read_sql(table_name, con=engine)

def to_db(df, database_filename, table_name, index = False):
    """
    Save a data frame as a SQLite DB file to the given location with the given table name
    :param df: The data frame to save
    :param database_filename: The DB file to create (NOTE: Will be replaced if it exists)
    :param index: (Optional, Default: False) Whether or not to create an index column in the saved table
    :param table_name: The name of the table to contain the data frame data
    """

    # If the DB file exists, delete it
    if os.path.exists(database_filename):
        os.remove(database_filename)

    # Save data to an sqlite db
    engine = sq.create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=index)

def one_hot_encode(df, column_name, prefix = '', replace_column = True, insert_to_end = False):
    """
    Performs one hot encoding on the given column in the data and replaces this column with the
    new one hot encoded columns
    :param df: The data frame in question
    :param column_name: The column to one hot encode
    :param prefix: (Optional, Default: column_name) The prefix for the new columns
    :param replace_column: (Optional, Default: True) Whether or not to replace the column to encode
    :param insert_to_end: (Optional, Default: False) Whether or not to add encoded columns at the end
    :return: The same data frame with the specified changes
    """

    dummies_insertion_index = df.columns.get_loc(column_name)
    dummies = pd.get_dummies(df[column_name], prefix=column_name if prefix == '' else prefix)

    if replace_column:
        df = df.drop([column_name], axis=1)
    else:
        dummies_insertion_index += 1

    if insert_to_end:
        df = pd.concat([df, dummies], axis=1)
    else:
        for column_to_insert in dummies.columns:
            df.insert(loc=dummies_insertion_index, column=column_to_insert, value=dummies[column_to_insert])
            dummies_insertion_index += 1

    return df

def normalize_confusion_matrix(cm_df):
    """
    Normalize the values in a confusion matrix to be between 0 and 1
    :param corr_df: The dataframe of the conusion matrix
    :return: The normalized matrix
    """

    for col in cm_df.columns:
        cm_df[col] = cm_df[col].apply(lambda x: x / cm_df[col].sum())

    return cm_df

def plot_scatter(data_frame, x, y, x_label = '', y_label = ''):
    """
    Plot a scatter plot given the data frame
    :param data_frame: The data frame to use for the scatter plot
    :param x: The column name for the x-axis
    :param y: The column name for the y-axis
    :param x_label: The label of the x-axis
    :param y_label: The label of the y-axis
    """

    x_label = x if x_label == '' else x_label
    y_label = y if y_label == '' else y_label

    data_frame = data_frame.dropna()

    standardize_plot_fonts()

    df_plot = pd.DataFrame()
    df_plot[x] = data_frame[x]
    df_plot[y] = data_frame[y]

    plot = df_plot.plot.scatter(x = x, y = y)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plot.set_title(y_label + ' vs. ' + x_label)

    plt.show()

def pad(ser, result_len, default_val = np.nan):
    """
    Pad a Series with values at the end to make it the length provided. Default padding is NaN
    :param ser: The Series
    :param result_len: The resulting length. This should be more than the current length of the series
    :param default_val: The value to pad with
    :return: The padded Series
    """

    if ser.size > result_len:
        raise ValueError('Result length ' + str(result_len) + ' needs to be more than ' + str(ser.size))

    return ser.reset_index(drop=True).reindex(range(result_len), fill_value=default_val)

def row_count(dataframe):
    """
    Gets the number of rows in a dataframe (most efficient way)
    :param dataframe: The dataframe to get the rows of
    :return: The row count
    """

    return len(dataframe.index)

def describe_hist(histogram, title, x_label, y_label):
    """
    Syntactic sugar to label the histogram axes and title
    :param histogram: The histogram
    :param title: The title to set
    :param x_label: The x-axis label to set
    :param y_label: The y-axis label to set
    """

    for ax in histogram.flatten():
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

def standardize_plot_fonts():
    """
    Standardize the title and axis fonts (Defaults to Title: 22, Axes: 15)
    """

    plt.rc('axes', labelsize=15) # Axis Font
    plt.rc('axes', titlesize=22) # Title Font

def whats(thing) :
    """
    Prints the type of object passed in
    Parameters:
        thing (Object): The object for which the type needs to be printed
    """

    print(type(thing))

def is_nan(value):
    """
    Returns true if value is NaN, false otherwise
    Parameters:
         value (Object): An object to test
    """

    return value != value

def read_csv(file_path, verbose=True):
    """
    Reads a csv file and returns the smallest possible dataframe
    :param file_path: The file path
    :param verbose: Whether or not to be verbose about the memory savings
    :return: An optimized dataframe
    """

    ret_val = pd.read_csv(file_path)
    return reduce_mem_usage(ret_val, verbose)

def reduce_mem_usage(df, verbose=True):
    """
    Takes a dataframe and returns one that takes the least memory possible.
    This works by going over each column and representing it with the smallest possible data structure.
    Example usage: my_data = pd.read_csv('D:/SomeFile.csv').pipe(reduce_mem_usage)
    Source: (https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65)
    Parameters:
        df (DataFrame): The dataframe to optimize
        verbose (bool): Whether or not to be verbose about the savings
    """

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

#region Properties

google_word2vec_model = None

word2vec_cache = None

#endregion

#region Private

def __get_confirm_token__(response):
    """
    Get a confirmation token from Google Drive (that says I'm ok with not scanning for viruses)
    :param response: The HTTP response object
    :return: The token
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def __save_response_content__(response, output_file_name):
    """
    Given an HTTP response object and a output file name, save the content to the file
    :param response: The HTTP response object
    :param output_file_name: The path of the output file
    """

    CHUNK_SIZE = 32768
    file_size = int(response.headers.get('Content-Length')) if response.headers.get('Content-Length') else None

    with open(output_file_name, "wb") as f:
        i = 1
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                mb_sofar = CHUNK_SIZE * i / 1024 / 1024
                if file_size:
                    percentage = (CHUNK_SIZE * i / file_size * 100)
                    sy.stdout.write('\r' + '[                                                  ]'
                                     .replace(' ', ':', int(percentage / 2)) + ' ' + str(
                        min(int(percentage), 100)) + '% (' + str(round(mb_sofar, 2)) + 'MB)')
                else:
                    sy.stdout.write('\r' + 'Unknown file size. ' + str(round(mb_sofar, 2)) + 'MB downloaded')
                f.write(chunk)
                i += 1
    print('')

#endregion