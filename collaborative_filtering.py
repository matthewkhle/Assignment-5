#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
df = np.array(df)

similar_users = []

vec2_temp = np.array(df[99])

vec2_temp = np.delete(vec2_temp, 0)
vec2_temp = np.delete(vec2_temp, 0)
vec2_temp = np.delete(vec2_temp, 2)

vec2 = []
vec2.append(vec2_temp)
vec2_average = np.average(vec2_temp)

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
for i in range(len(df) - 1):
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
   vec1_temp = np.array(df[i])
   user = vec1_temp[0]
   vec1_temp = np.delete(vec1_temp, 0)
   
   for x in range(len(vec1_temp)):
      vec1_temp[x] = float(vec1_temp[x])

   average = np.average(vec1_temp)
   galleries = vec1_temp[0]
   juice_bars = vec1_temp[3]

   vec1_temp = np.delete(vec1_temp, 0)
   vec1_temp = np.delete(vec1_temp, 2)
   # print(vec2_temp)
   vec1 = []
   vec1.append(vec1_temp)
   similarity = cosine_similarity(vec1, vec2)
   similar_users.append(tuple([user, similarity, average, galleries, juice_bars]))

#find the top 10 similar users to the active user according to the similarity calculated before
#--> add your Python code here
# similar_users.sort(key=lambda, tup: tup[1])
similar_users = sorted(similar_users, key=lambda x:x[1], reverse=True)

#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries (0) and juice bars? (2))
#--> add your Python code here

total_similarities = 0
for i in similar_users[:10]:
   total_similarities += i[1]

galleries_numerator = 0
for i in similar_users[:10]:
   galleries_numerator = galleries_numerator + (i[1] * (i[3] - i[2]))

galleries_recommendation = vec2_average + galleries_numerator / total_similarities
print("Galleries Prediction: "+ str(galleries_recommendation))

juice_bars_numerator = 0
for i in similar_users[:10]:
   juice_bars_numerator = juice_bars_numerator + (i[1] * (i[4] - i[2]))

juice_bars_recommendation = vec2_average + juice_bars_numerator / total_similarities
print("Juice Bars Prediction: "+ str(juice_bars_recommendation))
