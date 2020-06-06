import utility.util as ut
import pandas as pd
import sections as se
import investigations as iv
import numpy as np

se.widen_df_display()

articles = ut.read_csv('data/articles.csv')
interactions = ut.read_csv('data/interactions.csv')

# ##################################################################################
# #   Part I: Exploratory Data Analysis
# ##################################################################################
#
# # SECTION 1.1
# max_views_by_user = se.get_max_views_by_user(interactions)
# # se.show_num_article_interaction_distribution(interactions) UNCOMMENT ME BEFORE SUBMISSION!!
#
# # SECTION 1.2
# print(f'median: {se.get_median_num_article_interaction(interactions)}')
# print(f'max views by user: {se.get_max_num_article_interaction(interactions)}')
#
# # SECTION 1.3
# articles = se.remove_dupes(articles)
# articles.to_csv('data/articles.csv', index=False)
#
# # SECTION 1.4
# unique_articles = se.get_num_articles_with_interaction(interactions)
# total_articles = se.get_num_articles(articles)
# unique_users = se.get_unique_users(interactions)
# user_article_interactions = len(interactions)
#
# # SECTION 1.5
# # max_views_by_user =
# most_viewed_article_id = se.get_most_viewed_article_id(interactions)
# most_viewed_article_id_frequency = se.get_most_viewed_article_views(interactions)
#
# # SECTION 1.6
# print(f'50% of individuals have {se.get_median_num_article_interaction(interactions)} or fewer interactions.')
# print(f'The total number of user-article interactions in the dataset is {user_article_interactions}.')
# print(f'The maximum number of user-article interactions by any 1 user is {max_views_by_user}.')
# print(f'The most viewed article in the dataset was viewed {most_viewed_article_id_frequency} times.')
# print(f'The article_id of the most viewed article is {most_viewed_article_id}.')
# print(f'The number of unique articles that have at least 1 rating {unique_articles}.')
# print(f'The number of unique users in the dataset is {unique_users}')
# print(f'The number of unique articles on the IBM platform: {total_articles}')


# ##################################################################################
# #   Part II: Rank-Based Recommendations
# ##################################################################################
#
# # SECTION 2.1
# print(se.get_top_articles(10, interactions))
# print(se.get_top_article_ids(10, interactions))
#
# # SECTION 2.2
# top_5 = se.get_top_articles(5, interactions)
# top_10 = se.get_top_articles(10, interactions)
# top_20 = se.get_top_articles(20, interactions)
# print(top_5)
# print(top_10)
# print(top_20)

##################################################################################
#   Part III: User-User Based Collaborative Filtering
##################################################################################

# SECTION 3.1
# user_item_matrix = se.create_user_item_matrix(interactions) UNCOMMENT ME BEFORE SUBMISSION!!
# user_item_matrix.to_csv('data/user_item_matrix.csv', index=False)
# se.test_user_item_matrix(user_item_matrix)

user_item_matrix = ut.read_csv('data/user_item_matrix.csv')

# SECTION 3.2
# Do a spot check of your function
print(f'The 10 most similar users to user 1 are: {se.find_similar_users(1, user_item_matrix)[:10]}')
print(f'The 5 most similar users to user 3933 are: {se.find_similar_users(3933, user_item_matrix)[:5]}')
print(f'The 3 most similar users to user 46 are: {se.find_similar_users(46, user_item_matrix)[:3]}')

# SECTION 3.3
article_ids = se.user_user_recs(1, 10, user_item_matrix, interactions)
print(f'The following are the recommended article ids for user id 1: {article_ids}')

# SECTION 3.4
se.test_get_articles(user_item_matrix, interactions)

# SECTION 3.5
# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = se.user_user_recs_part2(20, 10, user_item_matrix, interactions)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)

# SECTION 3.6
user1_most_sim = se.find_similar_users(1, user_item_matrix)[0]
user131_10th_sim = se.find_similar_users(131, user_item_matrix)[10]
print(f'The user that is most similar to user 1.: {user1_most_sim}')
print(f'The user that is the 10th most similar to user 131: {user131_10th_sim}')

# SECTION 3.7
# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to
new_user_recs = se.get_top_article_ids(10, interactions)
assert set(map(lambda x: str(x), new_user_recs)) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."
print("That's right!  Nice job!")

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

