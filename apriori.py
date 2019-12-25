#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:05:29 2019

@author: fethi
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

veriler = pd.read_csv('GroceryStoreDataSet.csv', names=['products'], header=None)

# print(veriler)
# print(veriler.columns)
# print("Values",veriler.values)

data = list(veriler["products"].apply(lambda x: x.split(',')))
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
print(te.columns_)
df = pd.DataFrame(te_data, columns=te.columns_)

df1 = apriori(df, min_support=0.01, use_colnames=True)
# print(df1.sort_values(by="support",ascending=False))
df1['length'] = df1['itemsets'].apply(lambda x: len(x))
# print(df1[(df1['length']==2) & (df1['support']>=0.05)])
