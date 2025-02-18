import utility.util as ut
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
import models.extensions as ex

def print_dicussion():
    """
    Prints my final thoughts on the analysis
    """

    print('\n\nFirst of all, the plot seems to indicate a fairly straight-forward conclsion '
          '(ie. the more latent features we use, the better the accuracy).\nThe fact that we have trained '
          'and tested on separate data lends further credibility to this conclusion, however, the fact that '
          'there seems to be a \nmathematical relationship between the accuracy and the number of latent '
          'features also indicates over-fitting because this relationship is likely a \nside-effect of SVD '
          '(as this is a deterministic technique as opposed to a stochastic one)\n\nIf I were to productionize '
          'this, I would take the following steps:\n- First of all I would have invested effort in content-based '
          'recommendation. I believe it should be possible to create effective models by combining \nword2vec and neural'
          ' networks. For eg. vectorizing articles by taking the average of all the word vectors and then recommending'
          ' articles whose word vector \ndistances are small\n- I would factor in the number of times a user has '
          'interacted with an article and assign weights to every unique user-article combo.\n- This is more'
          ' a customer journey related suggestion, but it would be a good idea (when on-boarding users) to get them '
          'to select a list of article \ncategories that interest them and categorize users that way.'
          ' This would go some way to mitigate the cold start problem.')

def plot_accuracy_vs_latent_features_train_test(u_train, s_train, vt_train, user_item_matrix_test):
    """
    Plots how the number of latent features affects our ability to predict user item interaction BUT based on the
    results of the decomposition of the train user item matrix vs the test user item matrix
    :param u_train: The unit vector component of the training side
    :param s_train: The latent feature effectiveness diagonal matrix of the training side
    :param vt_train: V-Transpose - The matrix of latent features to items of the training side
    :param user_item_matrix_test: The test user item interaction matrix
    """

    user_item_matrix_test = user_item_matrix_test[user_item_matrix_test.columns[1:]].values.astype(int)
    num_latent_features = np.arange(10, 800, 20)
    errors = []

    for k in num_latent_features:

        # Prune
        u_pruned, s_pruned, vt_pruned = u_train[:, :k], np.diag(s_train[:k]), vt_train[:k, :]
        user_item_prediction = np.round(np.dot(np.dot(u_pruned, s_pruned), vt_pruned))
        diffs = user_item_matrix_test - user_item_prediction

        # Calculate error fraction
        err = np.sum(np.abs(diffs))
        err /= diffs.shape[0] * diffs.shape[1]
        errors.append(err)

    plt.plot(num_latent_features, 1 - np.array(errors))
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Latent Features')

    plt.show()

def create_test_and_train_user_item(interactions_train, interactions_test):
    """
    Create user item matrices from the train and test split of the raw interactions data
    :param interactions_train: The train section of the raw interactions data
    :param interactions_test: The test section of the raw interactions data
    :return:
        The train user item matrices
        The test user item matrices
        The user ids that appear in both train and test
        The article ids that appear in both train and test
    """

    user_item_matrix_train = create_user_item_matrix(interactions_train)
    user_item_matrix_test = create_user_item_matrix(interactions_test)

    train_user_ids = user_item_matrix_train['user_id']
    test_user_ids = user_item_matrix_test['user_id']
    intersect_user_ids = set(train_user_ids).intersection(set(test_user_ids))

    train_articles_ids = user_item_matrix_train.columns[1:]
    test_article_ids = user_item_matrix_test.columns[1:]
    intersect_article_ids = np.intersect1d(train_articles_ids, test_article_ids)

    return user_item_matrix_train, user_item_matrix_test, intersect_user_ids, intersect_article_ids

def plot_accuracy_vs_latent_features(interactions, user_item_arr_matrix, u, s, vt):
    """
    Given the user item array matrix and the matrices resulting from the decomposition of the user item matrix, plots
    a chart to show how the number of latent features affects the accuracy of prediction
    :param interactions: The raw interactions data
    :param user_item_arr_matrix: The user item interaction matrix
    :param u: The unit vector component
    :param s: The latent feature effectiveness diagonal matrix
    :param vt: V-Transpose - The matrix of latent features to items
    """

    num_latent_feats = np.arange(10, 800, 20)
    errors = []

    for k in num_latent_feats:
        # restructure with k latent features
        s_pruned, u_pruned, vt_pruned = np.diag(s[:k]), u[:, :k], vt[:k, :]

        # take dot product
        user_item_est = np.around(np.dot(np.dot(u_pruned, s_pruned), vt_pruned))

        # compute error for each prediction to actual value
        diffs = np.subtract(user_item_arr_matrix, user_item_est)

        # total errors and keep track of them
        err = np.sum(np.sum(np.abs(diffs)))
        errors.append(err)

    plt.plot(num_latent_feats, 1 - np.array(errors) / interactions.shape[0])
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Latent Features')

    plt.show()

def perform_svd(user_item_matrix, user_ids = None, article_ids = None):
    """
    Perform SVD on the user item interaction matrix and return the results
    :param user_item_matrix: The user item interaction matrix
    :param user_ids: The user ids to preserve (Default: All of them)
    :param article_ids: The article ids to preserve (Default: All of them)
    :return: The U, Σ, V-transpose matrices
    """

    # If both values are provided, prune to preserve subset
    if user_ids is not None and article_ids is not None:
        article_ids = np.intersect1d(article_ids, user_item_matrix.columns.values[1:])
        user_item_matrix = user_item_matrix[user_item_matrix['user_id'].isin(user_ids)]
        user_item_matrix = user_item_matrix[article_ids].values.astype(int)
    else:
        user_item_matrix = user_item_matrix[user_item_matrix.columns[1:]].values.astype(int)

    # Perform SVD
    u, s, vt = np.linalg.svd(user_item_matrix, False)

    return user_item_matrix, u, s, vt

def get_top_sorted_users(user_id, user_item_matrix, interactions):
    """
    Finds the most similar users in terms of the browsing habits (Ranks users who interact more, higher)
    :param user_id: The user Id to which we have to find the similar users
    :param user_item_matrix: The user item interaction matrix
    :param interactions: The raw interaction data
    :return: A list of user ids (excluding the queried one) that ranks users from most similar to least similar
    """

    # Get interaction vectors
    user_vector = user_item_matrix[user_item_matrix['user_id'] == user_id].iloc[0].tolist()[1:]

    # Add user id column to raw interaction data
    email_to_user_id, user_id_column = get_email_to_id_mapping(interactions)
    interactions['user_id'] = user_id_column

    # Add similarity and interaction score columns
    user_item_matrix['similarity'] = user_item_matrix.apply(lambda row: np.dot(user_vector, row.tolist()[1:]), axis=1)
    user_item_matrix['interaction_score'] = user_item_matrix['user_id'].apply(lambda user_id: len(interactions[interactions['user_id'] == user_id]))

    # Sort by new columns and return user id
    similar_users = user_item_matrix.sort_values(by=['similarity', 'interaction_score'], ascending=False)['user_id']

    # Return all except for original user id
    return similar_users[similar_users != user_id]


def user_user_recs_part2(user_id, m, user_item_matrix, interactions):
    """
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    :param user_id: The user id
    :param m: The m top recommendations to get
    :param user_item_matrix: The user item interaction
    :param interactions: The raw interaction data
    :return: The top m recommendations
    """

    seen_article_ids, seen_article_names = get_user_articles(user_id, user_item_matrix, interactions)
    similar_users_ids = get_top_sorted_users(user_id, user_item_matrix, interactions)
    recommended_article_ids = []
    for similar_users_id in similar_users_ids:
        # Get similar articles
        similar_article_ids, similar_article_names = get_user_articles(similar_users_id, user_item_matrix, interactions)
        # Find the unseen ones
        unseen = np.setdiff1d(similar_article_ids, seen_article_ids)
        # Make them seen
        seen_article_ids = np.concatenate((seen_article_ids, unseen), axis=None)
        # Add them to recommendations
        recommended_article_ids = np.concatenate((recommended_article_ids, unseen), axis=None)
        # Break if we have enough
        if len(recommended_article_ids) >= m:
            break

    # Sort ids by number of interactions and then prune lowest
    recommended_article_ids = ex.dictionary((article_id,
                            ut.row_count(interactions[interactions['article_id'] == float(article_id)]))
                                for article_id in recommended_article_ids).get_sorted().key_list()[0:m]

    # Get article names
    recommended_article_names = get_article_names(recommended_article_ids, interactions)

    return recommended_article_ids, recommended_article_names

def test_get_articles(user_item_matrix, interactions):
    """
    Test the get article functionality
    :param user_item_matrix: The user item interaction
    :param interactions: The raw interaction data
    """

    print('-----BEGIN TEST-----')

    # Test your functions here - No need to change this code - just run this cell
    assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'], interactions)) == set(
        ['using deep learning to reconstruct high-resolution audio',
         'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model',
         'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding',
         'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
    assert set(get_article_names(['1320.0', '232.0', '844.0'], interactions)) == set(
        ['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery',
         'use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
    assert set(get_user_articles(20, user_item_matrix, interactions)[0]) == set(['1320.0', '232.0', '844.0'])
    assert set(get_user_articles(20, user_item_matrix, interactions)[1]) == set(
        ['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery',
         'use the cloudant-spark connector in python notebook'])
    assert set(get_user_articles(2, user_item_matrix, interactions)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
    assert set(get_user_articles(2, user_item_matrix, interactions)[1]) == set(['using deep learning to reconstruct high-resolution audio',
                                                'build a python app on the streaming analytics service',
                                                'gosales transactions for naive bayes model',
                                                'healthcare python streaming application demo',
                                                'use r dataframes & ibm watson natural language understanding',
                                                'use xgboost, scikit-learn & ibm watson machine learning apis'])
    print("If this is all you see, you passed all of our tests!  Nice job!")

    print('-----END TEST-----')

def get_article_names(article_ids, interactions):
    """
    Get the article names (title) for the given article ids
    :param article_ids:
    :param interactions:
    :return:
    """

    return [interactions[interactions['article_id'] == float(article_id)].iloc[0]['title']
            for article_id in article_ids]


def get_user_articles(user_id, user_item_matrix, interactions):
    """
    Gets a list of article ids and article titles that have been seen by the given user id
    :param user_id: The user id
    :param user_item_matrix: The interaction matrix
    :param interactions: The raw interaction data
    :return: 2 outputs: article ids and article names
    """

    all_article_ids = user_item_matrix.columns.values
    all_article_ids = all_article_ids[all_article_ids != 'user_id']

    article_ids = all_article_ids[
        user_item_matrix[user_item_matrix['user_id'] == user_id].iloc[0][all_article_ids] == 1
    ]

    article_names = get_article_names(article_ids, interactions)

    return article_ids, article_names


def user_user_recs(user_id, m, user_item_matrix, interactions):
    """
    For the given user id, recommend m articles. [These are the first m articles found (un-seen by the user) when
    looking for articles similar users interacted with]
    :param user_id: The user id for whom to recommend articles
    :param m: The number of articles to recommend
    :param user_item_matrix: The interaction matrix
    :param interactions: The raw interaction data
    :return: The article ids recommended
    """

    seen_article_ids, seen_article_names = get_user_articles(user_id, user_item_matrix, interactions)
    similar_users_ids = find_similar_users(user_id, user_item_matrix)

    recommended_article_ids = []
    for similar_users_id in similar_users_ids:
        similar_article_ids, similar_article_names = get_user_articles(similar_users_id, user_item_matrix, interactions)
        unseen = np.setdiff1d(similar_article_ids, seen_article_ids)
        recommended_article_ids = np.unique(np.concatenate((recommended_article_ids, unseen), axis=None))
        if len(recommended_article_ids) >= m:
            break

    return recommended_article_ids[0:m]

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
    email_to_id_mapping, user_id_column = get_email_to_id_mapping(interactions)
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

    return interactions.groupby(r'article_id').count().sort_values(by='email', ascending=False).index.values[0:n]

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

def get_max_views_by_user(interactions):
    """
    Get the maximum number of times an article has been viewed by a user
    :param interactions: The user article interaction data
    :return: The maximum number of times an article has been viewed by a user
    """

    return interactions.groupby('email')['article_id'].count().max()

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

    return coded_dict, email_encoded

def widen_df_display():
    """
    Widens the way dataframes are printed (setting lifetime is runtime)
    """

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 100)

#endregion