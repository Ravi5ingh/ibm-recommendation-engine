import utility.util as ut
import statistics as st
import models.extensions as ex
import sections as se
import numpy as np
import pandas as pd

se.widen_df_display()

se.print_dicussion()

#region Old

# user_item_matrix = ut.read_csv('data/user_item_matrix.csv')
# user_item_matrix = user_item_matrix[user_item_matrix.columns[1:]].values.astype(int)

# print(u.shape)

# print(s.shape)
# print(vt.shape)

# Get values of dictionary in sorted format
# list(map(lambda x: x[1], sorted(email_interactions.items(), key=lambda x: x[1])))

# interactions = ut.read_csv('data/interactions.csv')
# articles = ut.read_csv('data/articles.csv')
#
# user_item_matrix = ut.read_csv('data/user_item_matrix.csv')

# se.find_similar_users_int(1, user_item_matrix)

# print(se.get_top_sorted_users(1, user_item_matrix, interactions))

# print(se.user_user_recs(1, 10, user_item_matrix, interactions))

# se.test_get_articles(user_item_matrix, interactions)

# print(se.user_user_recs_part2(1, 10, user_item_matrix, interactions))

# Quick spot check - don't change this code - just use it to test your functions
# rec_ids, rec_names = se.user_user_recs_part2(20, 10, user_item_matrix, interactions)
# print("The top 10 recommendations for user 20 are the following article ids:")
# print(rec_ids)
# print()
# print("The top 10 recommendations for user 20 are the following article names:")
# print(rec_names)

#endregion