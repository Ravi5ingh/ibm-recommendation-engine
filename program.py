import utility.util as ut
import pandas as pd
import sections as se
import investigations as iv
import numpy as np

se.widen_df_display()

articles = ut.read_csv('data/articles.csv')
interactions = ut.read_csv('data/interactions.csv')

##################################################################################
#   Part I: Exploratory Data Analysis
##################################################################################

# SECTION 1.1
max_views_by_user = se.get_max_views_by_user(interactions)
# se.show_num_article_interaction_distribution(interactions) UNCOMMENT ME BEFORE SUBMISSION!!

# SECTION 1.2
print(f'median: {se.get_median_num_article_interaction(interactions)}')
print(f'max views by user: {se.get_max_num_article_interaction(interactions)}')

# SECTION 1.3
articles = se.remove_dupes(articles)
articles.to_csv('data/articles.csv', index=False)

# SECTION 1.4
unique_articles = se.get_num_articles_with_interaction(interactions)
total_articles = se.get_num_articles(articles)
unique_users = se.get_unique_users(interactions)
user_article_interactions = len(interactions)

# SECTION 1.5
# max_views_by_user =
most_viewed_article_id = se.get_most_viewed_article_id(interactions)
most_viewed_article_id_frequency = se.get_most_viewed_article_views(interactions)

# SECTION 1.6
print(f'50% of individuals have {se.get_median_num_article_interaction(interactions)} or fewer interactions.')
print(f'The total number of user-article interactions in the dataset is {user_article_interactions}.')
print(f'The maximum number of user-article interactions by any 1 user is {max_views_by_user}.')
print(f'The most viewed article in the dataset was viewed {most_viewed_article_id_frequency} times.')
print(f'The article_id of the most viewed article is {most_viewed_article_id}.')
print(f'The number of unique articles that have at least 1 rating {unique_articles}.')
print(f'The number of unique users in the dataset is {unique_users}')
print(f'The number of unique articles on the IBM platform: {total_articles}')


# titles = se.get_top_articles(10, interactions)
#
# for title in titles:
#       print(title)
#
# ids = se.get_top_article_ids(10, interactions)
#
# print(ids)

# se.show_num_article_interaction_distribution(interactions)

# print(se.get_median_num_article_interaction(interactions))

# old, new = se.get_max_num_article_interaction(interactions)
#
# print(old)
# print(new)

# print(se.get_most_viewed_article_id(interactions))

# print(se.get_most_viewed_article_views(interactions))

# user_item_matrix = se.create_user_item_matrix(interactions)

# user_item_matrix.to_csv('data/user_item_matrix.csv', xindex=False)

# user_item_matrix = pd.read_csv('data/user_item_matrix.csv')

# se.test_user_item_matrix(user_item_matrix)

# print(se.find_similar_users(1243, user_item_matrix))

# se.test_similar_users(user_item_matrix)

# print(se.get_article_names([1430, 1276], interactions))

# article_ids, article_names = se.get_user_articles(1, user_item_matrix, interactions)
#
# article_ids = se.user_user_recs(1, 10, user_item_matrix, interactions)

#print(article_ids)

# user_item_arr_matrix, u, s, vt = se.perform_svd(user_item_matrix)
#
# se.plot_accuracy_vs_latent_features(interactions, user_item_arr_matrix, u, s, vt)

# interactions_train = interactions.head(40000)
# interactions_test = interactions.tail(5993)
#
# user_item_matrix_train, user_item_matrix_test, intersect_user_ids, intersect_article_ids = \
#     se.create_test_and_train_user_item(interactions_train, interactions_test)
#
# # print(f'user_item_matrix_train is {len(user_item_matrix_train)} by {len(user_item_matrix_train.columns)}')
# # print(f'user_item_matrix_test is {len(user_item_matrix_test)} by {len(user_item_matrix_test.columns)}')
# # print(f'test_user_ids contains {len(test_user_ids)} items')
# # print(f'test_article_ids contains {len(test_article_ids)} items')
#
# user_item_matrix_train, u_train, s_train, vt_train = se.perform_svd(user_item_matrix_train, intersect_user_ids, intersect_article_ids)
#
# se.plot_accuracy_vs_latent_features_train_test(u_train, s_train, vt_train, user_item_matrix_test)

