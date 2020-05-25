import utility.util as ut
import pandas as pd
import sections as se
import investigations as iv
import utility.processor as pr

se.widen_df_display()

articles = ut.read_csv('data/articles.csv')
interactions = ut.read_csv('data/interactions.csv')

# print(se.get_median_num_article_interaction(interactions))

# print(se.get_max_num_article_interaction(interactions))

# articles = se.remove_dupes(articles)
#
# articles.to_csv('data/articles.csv', index=False)

print(f'{se.get_num_articles_with_interaction(interactions)} '
      f'articles have been interacted with {len(interactions)} times')

print(f'Number of unique articles: {se.get_num_articles(articles)}')

print(f'Number of unique users: {se.get_unique_users(interactions)}')

print(f'Number of unique user to article interactions: {se.get_unique_user_article_interactions(interactions)}')

article_interaction_frequency = pr.get_article_to_interactions_mapping(interactions)
most_viewed_article_id = article_interaction_frequency.get_sorted(ascending=False).key_at(0)
most_viewed_article_id_frequency = article_interaction_frequency.value_at(0)
print(f'The most viewed article is article id: {most_viewed_article_id}. '
      f'It was viewed {most_viewed_article_id_frequency} times')