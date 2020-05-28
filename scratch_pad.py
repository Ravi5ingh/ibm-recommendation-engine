import utility.util as ut
import statistics as st
import models.extensions as ex
import sections as se
import numpy as np

se.widen_df_display()

# Get values of dictionary in sorted format
# list(map(lambda x: x[1], sorted(email_interactions.items(), key=lambda x: x[1])))

interactions = ut.read_csv('data/interactions.csv')

x = [1, 1, 1, 0, 0, 0, 0]
y = [1, 1, 0, 0, 1, 1, 0]

print(np.dot(x, y))