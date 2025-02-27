# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)

# prepare the data for the priori 

X = dataset.iloc[:, :].values
# print(X)


transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training Apriori on the dataset

from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

# Visualizing the results
results=list(rules)
# print(results)


the_rules = []
for result in results:
    the_rules.append({'rule': ','.join(result.items),'support':result.support,'confidence':result.ordered_statistics[0].confidence,'lift':result.ordered_statistics[0].lift})
df = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])
# print(df)
