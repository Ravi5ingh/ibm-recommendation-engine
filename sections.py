import utility.util as ut
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
import models.extensions as ex

def test_similar_users(user_item_matrix):
    """
    Perform some sanity checking on the find similar users functionality
    :param user_item_matrix: The user item interaction matrix
    """

    # Do a spot check of your function
    print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1, user_item_matrix)[:10]))
    print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933, user_item_matrix)[:5]))
    print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46, user_item_matrix)[:3]))

def find_similar_users(user_id, user_item_matrix):
    """
    Finds the most similar users in terms of the browsing habits
    :param user_id: The user Id to which we have to find the similar users
    :param user_item_matrix: The user item interaction matrix
    :return: A list of user ids (excluding the queried one) that ranks users from most similar to least similar
    """

    user_vector = user_item_matrix[user_item_matrix['user_id'] == user_id].iloc[0].tolist()[1:]
    other_users_vectors = user_item_matrix[user_item_matrix['user_id'] != user_id]

    user_similarity = ex.dictionary()
    for index, row in other_users_vectors.iterrows():
        row_list = row.tolist()
        user_similarity[row_list[0]] = np.dot(user_vector, row_list[1:])

    user_similarity = user_similarity.get_sorted(ascending=False)

    return list(user_similarity.keys())

def test_user_item_matrix(user_item_matrix):
    """
    Unit tests for user item matrix data
    :param user_item_matrix: The user item matrix df
    """

    assert user_item_matrix.shape[0] == 5149, \
        'Oops!  The number of users in the user-article matrix doesn\'t look right.'

    assert user_item_matrix.shape[1] == 714 + 1, \
        'Oops!  The number of articles in the user-article matrix doesn\'t look right.'

    assert sum(user_item_matrix[user_item_matrix['user_id'] == 1].loc[0].tolist()[1:]) == 36, \
        'Oops!  The number of articles seen by user 1 doesn\'t look right.'

    print("You have passed our quick tests!  Please proceed!")

def create_user_item_matrix(interactions):
    """
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    :param interactions: The interactions data
    :return: The user matrix
    """

    # Create df with user_id column
    print('Creating User Id column...')
    user_item_matrix = pd.DataFrame()
    email_to_id_mapping = get_email_to_id_mapping(interactions)
    user_item_matrix['user_id'] = email_to_id_mapping.values()

    # Create df with zeros for each article_id
    print('Creating df with zeros...')
    unique_article_ids = set(interactions['article_id'])
    article_df = pd.DataFrame(columns=unique_article_ids)
    user_id_count = ut.row_count(user_item_matrix)
    current = 1
    total = len(article_df.columns)
    for column in article_df.columns:
        article_df[column] = np.zeros(user_id_count)
        ut.update_progress(current, total)
        current += 1

    # Join both dfs
    print('Joining...')
    user_item_matrix = user_item_matrix.join(article_df)

    # Flip switch to 1 for each unique interaction
    print('Getting unique interactions...')
    unique_interactions = set(interactions.apply(lambda row: str(row['article_id']) + '--' + str(row['email']), axis=1))

    current = 1
    total = len(unique_interactions)
    print('Flipping switches from 0 to 1...')
    for interaction in unique_interactions:
        sections = interaction.split('--')
        article_id = float(sections[0])
        email = sections[1] if sections[1] != 'nan' else np.nan
        user_id = email_to_id_mapping[email]
        user_item_matrix.loc[user_item_matrix['user_id'] == user_id, article_id] = 1
        ut.update_progress(current, total)
        current += 1

    return user_item_matrix

def get_email_to_id_mapping(interactions):
    """
    Map every email to a user id and return the dictionary mapping
    :param interactions: The interactions data
    :return: The dictionary mapping
    """

    return email_mapper(interactions)

def add_userid(interactions):
    """
    Adds a user Id column to the interactions data
    :param interactions: The interactions data
    :return: The augmented interactions dataframe
    """

    interactions['user_id'] = 0
    emails = set(interactions['email'])

    current_id = 1
    total = len(emails)
    for email in emails:
        interactions.loc[interactions['email'] == email, 'user_id'] = current_id
        ut.update_progress(current_id, total)
        current_id += 1


    return interactions


def get_top_article_ids(n, interactions):
    """
    Get the top most interacted with article ids
    :param n: The number of top article titles
    :param interactions: The interaction data
    :return: A list of the top 'n' article ids
    """

    return interactions.groupby(r'article_id').count().sort_values(by='email', ascending=False).index

def get_top_articles(n, interactions):
    """
    Get the top most interacted with article titles
    :param n: The number of top articles to return
    :param interactions: The interactions data
    :return: A list of the top 'n' article titles
    """

    titles = []
    title_indices = interactions.groupby('article_id').count().sort_values(by='email', ascending=False).index

    i = 0
    for index in title_indices:
        retrieved_title = interactions[interactions['article_id']==index]['title'].iloc[0]
        titles.append(retrieved_title)
        i += 1
        if i >= n:
            break

    return titles

def get_most_viewed_article_views(interactions):
    """
    Get the number of times the most viewed article was viewed
    :param interactions: The interactions data
    :return: The number of times the most viewed article was viewed
    """

    grouped = interactions.groupby('article_id').count()['email']
    return grouped[grouped.idxmax()]

def get_most_viewed_article_id(interactions):
    """
    Get the most viewed article id in the data set
    :param interactions: The interactions data
    :return: The most viewed article id
    """

    return interactions.groupby('article_id').count()['email'].idxmax()

def get_unique_user_article_interactions(interactions):
    """
    Get the number of unique user article interactions
    :param interactions: The interactions data
    :return: The number of unique interactions
    """

    return len(set(interactions.apply(lambda row: str(row['article_id']) + str(row['email']), axis=1)))

def get_unique_users(interactions):
    """
    Get the number of unique users in the interactions data
    :param interactions: The interactions data
    :return: The number of uniques users
    """

    return len(set(interactions['email']))

def get_num_articles(articles):
    """
    Get the number of unique articles
    :param articles: The articles data
    :return: The number of unique articles
    """

    return len(set(articles['article_id']))

def get_num_articles_with_interaction(interactions):
    """
    Get the unique number of articles that have an interaction with the user
    :param interactions: The interactions data
    :return: The number of articles
    """

    return len(set(interactions['article_id']))

def remove_dupes(articles):
    """
    Removes duplicate articles (based on article_id) and just retains the first instance
    :param articles: The articles data
    :return: The articles data with dupes removed
    """

    return articles.drop_duplicates(subset='article_id', keep='first')


def get_max_num_article_interaction(interactions):
    """
    Get the number of interactions of the most active user
    :param interactions: The user article interaction data
    :return: The maximum number of articles any user has interacted with
    """

    return max(interactions.groupby('email').count()['article_id'])

def get_median_num_article_interaction(interactions):
    """
    Get the median number of articles interacted with (50% of users interact with less than this number of articles)
    :param interactions: The user article interaction data
    :return: The median number articles interacted with
    """

    return st.median(interactions.groupby('email').count()['article_id'])


def show_num_article_interaction_distribution(interactions):
    """
    Visualizes the distribution of how many articles a user interacts with in the database
    :param interactions: The user article interaction data
    """

    email_interactions = interactions.groupby('email').count()['article_id']

    # Plot a histogram of the diff values
    plt.hist(email_interactions, bins=100)
    plt.title('Number of articles users interact with')
    plt.xlabel('Number of interactions')
    plt.ylabel('Email addresses with this number of interactions')

    plt.show()

def clean_raw_data():
    """
    Cleans the raw data by removing un-necessary columns
    """

    interactions = ut.read_csv('data/raw/user-item-interaction.csv')
    articles = ut.read_csv('data/raw/articles_community.csv')

    del interactions['Unnamed: 0']
    del articles['Unnamed: 0']
    del interactions['Unnamed: 0.1']
    del articles['Unnamed: 0.1']

    interactions.to_csv('data/interactions.csv', index=False)
    articles.to_csv('data/articles.csv', index=False)

#region Complementary

def email_mapper(interactions):
    """
    Default Code: just maps an email to a user id and returns the mapping
    :param interactions: The interaction data
    :return: The mapping
    """
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in interactions['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])

    return coded_dict

def widen_df_display():
    """
    Widens the way dataframes are printed (setting lifetime is runtime)
    """

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 100)

#endregion