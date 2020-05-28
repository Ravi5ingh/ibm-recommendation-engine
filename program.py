import utility.util as ut
import pandas as pd
import sections as se
import investigations as iv

se.widen_df_display()

# articles = ut.read_csv('data/articles.csv')
# interactions = ut.read_csv('data/interactions.csv')

# print(se.get_median_num_article_interaction(interactions))

# print(se.get_max_num_article_interaction(interactions))

# articles = se.remove_dupes(articles)
#
# articles.to_csv('data/articles.csv', index=False)

# print(f'{se.get_num_articles_with_interaction(interactions)} '
#       f'articles have been interacted with {len(interactions)} times')
#
# print(f'Number of unique articles: {se.get_num_articles(articles)}')
#
# print(f'Number of unique users: {se.get_unique_users(interactions)}')
#
# print(f'Number of unique user to article interactions: {se.get_unique_user_article_interactions(interactions)}')
#
# most_viewed_article_id = article_interaction_frequency.get_sorted(ascending=False).key_at(0)
# most_viewed_article_id_frequency = article_interaction_frequency.value_at(0)
# print(f'The most viewed article is article id: {most_viewed_article_id}. '
#       f'It was viewed {most_viewed_article_id_frequency} times')

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

user_item_matrix = pd.read_csv('data/user_item_matrix.csv')

# se.test_user_item_matrix(user_item_matrix)

# print(se.find_similar_users(1243, user_item_matrix))

se.test_similar_users(user_item_matrix)