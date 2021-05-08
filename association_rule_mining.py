#-------------------------------------------------------------------------
# AUTHOR: Matthew Le
# FILENAME: association_rule_mining.py
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #5
# TIME SPENT: 1.5 Hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
for index, row in df.iterrows():
    bread = 0
    wine = 0
    eggs = 0
    meat = 0
    cheese = 0
    pencil = 0
    diaper = 0
    milk = 0
    bagel = 0

    try:
        for item in row:
            # print(item)
            if 'Bread' == item:
                bread = 1
                
            if 'Wine' == item:
                wine = 1

            if 'Eggs' == item:
                eggs = 1

            if 'Meat' == item:
                meat = 1

            if 'Cheese' == item:
                cheese = 1

            if 'Pencil' == item:
                pencil = 1

            if 'Diaper' == item:
                diaper = 1
                
            if 'Milk' == item:
                milk = 1

            if 'Bagel' == item:
                bagel = 1

    except:
        print("title skipped")

    labels = {"Bread": bread, "Wine": wine, "Eggs": eggs, "Meat": meat, "Cheese": cheese, "Pencil": pencil, "Diaper": diaper, "Milk": milk, "Bagel": bagel}

    encoded_vals.append(labels)


#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

# print(ohe_df)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below
rules_df = pd.DataFrame(rules)
for index, row in rules_df.iterrows():

    print(str(row[0]) + " -> " + str(row[1]))
    print("Support: " + str(row[4]))
    print("Confidence: " + str(row[5]))



    #To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
    #use the gain formula provided right after.
    #prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
    #print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
    #-->add your python code below
    prior = min(rules['antecedent support'][i], rules['consequent support'][i])
    print("Prior: " + str(prior))
 
    print("Gain in Confidence: " + str(100*(row[5]-prior)/prior))
    print()

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()