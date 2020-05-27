import utility.util as ut
import statistics as st
import models.extensions as ex
import sections as se

se.widen_df_display()

# Get values of dictionary in sorted format
# list(map(lambda x: x[1], sorted(email_interactions.items(), key=lambda x: x[1])))

interactions = ut.read_csv('data/interactions.csv')

print(interactions.groupby('article_id').count().sort_values(by='email', ascending=False).index)