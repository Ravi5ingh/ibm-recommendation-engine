import utility.util as ut
import pandas as pd
import sections as se
import investigations as iv

se.widen_df_display()

articles = ut.read_csv('data/articles.csv')
interactions = ut.read_csv('data/interactions.csv')

# print(se.get_median_num_article_interaction(interactions))

# print(se.get_max_num_article_interaction(interactions))

articles = se.remove_dupes(articles)

articles.to_csv('data/articles.csv', index=False)


