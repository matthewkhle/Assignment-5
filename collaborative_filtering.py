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
# print(vec2_temp)
vec2 = []
vec2.append(vec2_temp)
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
   vec1_temp = np.delete(vec1_temp, 0)
   vec1_temp = np.delete(vec1_temp, 2)
   # print(vec2_temp)
   vec1 = []
   vec1.append(vec1_temp)
   similarity = cosine_similarity(vec1, vec2)
   similar_users.append(tuple([user, similarity]))

#find the top 10 similar users to the active user according to the similarity calculated before
#--> add your Python code here
# similar_users.sort(key=lambda, tup: tup[1])
similar_users = sorted(similar_users, key=lambda x:x[1], reverse=True)
print(similar_users[2][1])
#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
#--> add your Python code here