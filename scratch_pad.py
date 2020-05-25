import utility.util as ut
import statistics as st
import models.extensions as ex

# Get values of dictionary in sorted format
# list(map(lambda x: x[1], sorted(email_interactions.items(), key=lambda x: x[1])))

dd = ex.dictionary()

dd['a'] = 3
dd['b'] = 2
dd['c'] = 1

dd = dd.get_sorted()

print(dd)

