import utility.util as ut
import utility.processor as pr
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st


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
    Get the maximum number of articles any user has interacted with
    :param interactions: The user article interaction data
    :return: The maximum number of articles any user has interacted with
    """

    return max(pr.get_email_to_interactions_mapping(interactions).values())

def get_median_num_article_interaction(interactions):
    """
    Get the median number of articles interacted with (50% of users interact with less than this number of articles)
    :param interactions: The user article interaction data
    :return: The median number articles interacted with
    """

    return st.median(pr.get_email_to_interactions_mapping(interactions).values())


def show_num_article_interaction_distribution(interactions):
    """
    Visualizes the distribution of how many articles a user interacts with in the database
    :param interactions: The user article interaction data
    """

    email_interactions = pr.get_email_to_interactions_mapping(interactions)

    # Plot a histogram of the diff values
    plt.hist(email_interactions.values(), bins=1000)
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

def widen_df_display():
    """
    Widens the way dataframes are printed (setting lifetime is runtime)
    """

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 100)