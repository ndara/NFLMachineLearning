
# coding: utf-8

# # Presentation

# By: Nishanth Dara

# ## NFL Dataset Introduction

# This is a data set from Armchair Analysis composed of all stats from the last 15 years in the NFL. This includes team stats, players, records of every single play, defensive stats...the whole nine yards. I chose to explore the play by play scenarios in the NFL. Doing so I wanted to answer questions such as...
# * Is passing or running the ball more advantegeous?
# * Does the advantage depend on what down it is?
# * Do teams change gameplans depending on point differntial?
# 
# Ultimately the crux of this data analysis focuses on these 2 questions.
# * Can the yardage gained from an upcoming play be predicted by knowing the game clock, down, point differential, field position, and yards to go?
# * Using the same features, perhaps, can we predict whether a team will opt to pass or rush the ball?
# 
# This data analysis will be useful in many ways. Using these predictions will keep one a step above their friends on  Sundays. One can also use these predictions to win fantasy matchups. Coaches can use this analysis to change their gameplan on the fly based on what has worked well in the past. 

# ## Data Cleaning 

# Fortunately enough, the dataset for this project is polished and clean. There are rarely missing values in the tables I required for the project. The majority of my data cleaning dealt with modifying the dataset to fit my needs. For this project, I used 3 tables -- PASS, RUSH, and PLAY csvs. To prepare the data for exploration, I...
# * got rid of all unnecessary columns
# * changed columns names to be more understandable
# * created custom made columns based on multiple columns to reduce complexity
#     * i.e. changing offensive points and defensive points to a single point differential columns
# * modified the data types of each column appropriately
# 
# After several assert tests to ensure that my data was clean I rewrote these dataframes to a new csv.

# ### Imports

# For the project to be done efficiently, multiple Python libraries were utilized. Primarily Pandas, Seaborn, Scikit Learn were required for many aspects of the project.

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans


# In[2]:

pass_df = pd.read_csv('pass_df.csv', index_col=0)
rush_df = pd.read_csv('rush_df.csv', index_col=0)
play_df = pd.read_csv('play_df.csv', index_col=0)


# ## Exploration

# ### Distribution of Yardage Gained from Rushing Plays

# In[47]:

sns.distplot(rush_df.Yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage Gained of all Rushing Plays")
rush_yards = rush_df.Yards
print("mean: " + str(rush_yards.mean()) + " yards")
print("90% confidence interval: " + "( " + str(np.percentile(rush_yards,5)) + ", " + str(np.percentile(rush_yards,95)) + " )" )


# From this graph, we see that the mean yards gained from a rush play is 4.3 yards. Most running plays fall under the the interval of -2 to 14 yards. From the distplot, we see that the data is heavily centered near the mean.

# ### Compare With Distrubtions of Pass Plays

# In[41]:

sns.distplot(pass_df.Yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage of all Passing Plays")
pass_yards = pass_df.Yards
print("mean: " + str(pass_yards.mean()))
print("90% confidence interval: " + "( " + str(np.percentile(pass_yards,5)) + ", " + str(np.percentile(pass_yards,95)) + " )" )


# From this graph, we see that the mean yards gained from a pass play is 6.9 yards. Most pass plays fall under the the interval of 0 to 25 yards. From the distplot, we see that the data has a lot more variablity than the rush distribution. By looking at this graph, it seems to be that rushing the ball may be a better option. The reason for the extraordinary peak at 0 is the fact this accounts for incomplete passes. Let's observe the distribution for only complete passes.

# In[42]:

comp_yards = pass_df.loc[pass_df.Completion].Yards
sns.distplot(comp_yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage of all Completions")
print("mean: " + str(comp_yards.mean()))
print("90% confidence interval: " + "( " + str(np.percentile(comp_yards,5)) + ", " + str(np.percentile(comp_yards,95)) + " )" )


# This distribution removes the risk of a pass play being an incompletion. However, it allows us better to visualize how successful a pass is when it is successful. There is a mean of 11.5 yards, which in football is significant. A gain of 10+ yards usually implies a first down. We also see a much wider spread distribution of yards gained ranging from -5 all the way to 40+ yards. In the case that a pass play is completed, it surely surpassed the average yardage gained from a rushing play.

# ### Putting Rush and Pass Together

# In[43]:

pass_plays = pd.merge(play_df, pass_df, left_index=True, right_index=True, sort=True)
rush_plays = pd.merge(play_df, rush_df, left_index=True, right_index=True, sort=True)


# In[45]:

temp = play_df[(play_df.PtDiff <= 14) & (play_df.PtDiff >= -14)]
plt.figure(figsize=(40,20))
sns.countplot(x="PtDiff", hue="Type", data=temp)
plt.title("Type of Play vs. Point Differential")


# By simple observation of this data, we see that when teams are losing, they tend to pass the ball more. When a team is winning, they tend to rush the ball more. The peak at 0 is understandable because games are started with even scores so until a team scores all plays will be with done under a 0 point differential. Rushing the ball, although disadvantageous when trying to gain yards, is safer than passing risking an incompletion or worse, an interception. Clearly point differential is a valuable feature to use when attempting to predict whether a team will rush or pass the ball.

# In[46]:

temp = play_df[(play_df.YardsToGo <= 20)]
plt.figure(figsize=(30,10))
sns.countplot(x="YardsToGo", hue="Type", data=temp)
plt.title("Type of Play vs. Yards To Go")


# From these value count proportions, we see that teams that have fewer yards to go for the first down will run the ball. Teams that have more yards to go will throw the ball. This was expected because passing will generally return more yards than a rush play. The peak at 10 is understandable because the start of every new set of downs will begin with 10 yards to go. Yards to go also seems helpful in creating a very accurate predictor of whether a team will pass or run.

# ## Machine Learning

# For the machine learning portion of my project, I observed and created models to address the 2 questions of my project.
# * Can the yardage gained from an upcoming play be predicted by knowing the game clock, down, point differential, field position, and yards to go?
# * Using the same features, perhaps, can we predict whether a team will opt to pass or rush the ball?
# 
# Unfortunately, the yardage gained was highly unpredictable. After cross validating different models and changing hyperparameters I was still unable to have any accuracy in prediction. My r2_score was negative meaning it was better to randomly guess yardage gained than to use a model. From this we can understand that the beauty of sports is that it's unpredictable. Variables such as a defensive mistake, offensive ingenuity, referee mistake or even pure luck are all contributors to yardage gained and can't be accounted for.
# 
# On the bright side, models for predicting the type of play were more succesful. Let's observe them.

# ### Predicting Type of Play

# In[31]:

plays = pd.concat([pass_plays, rush_plays]).sort_index()


# Let's get rid of columns that we don't need any longer.

# In[32]:

delete_items = ['INT','Direction','Drive','GameID','Location','PTS','Sack','FUM']
for item in delete_items:
    del plays[item]
plays.head()


# We must also change the 'Type' to a integer value so that machine learning models will be able to use the column.

# In[33]:

plays.Type = plays.Type.str.replace('RUSH','0')
plays.Type = plays.Type.str.replace('PASS','1')
plays.Type = plays.Type.astype('int64')


# In[34]:

X = plays[['Down','YardsToGo','PtDiff','Time','Zone']]
y = plays['Type']


# X has our feature matrix. The features we are using include the 'Down', 'YardsToGo', 'PtDiff', 'Time', and field position ('Zone'). Y is the target array. We are attempting to predict the type of play the team will opt to. Let's proceed to split up the data to train and test using our models.

# In[35]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# We see right away that our model is working but not to the type of standard we would like. 64% accuracy is definitely better than guessing randomly but still not as high as we'd like it to be. Let's shift gears from a Gaussian model to a Random Forest Classifier.

# ### Using a Random Forest Classifier

# In[12]:

#THIS CODE TAKES A LONG TIME TO RUN
model = RandomForestClassifier(150)
model.fit(Xtrain, ytrain)


# In[13]:

#THIS CODE TAKES A LONG TIME TO RUN
y_pred = model.predict(Xtest)


# In[14]:

accuracy_score(ytest, y_pred)


# A random forest classifier is marginally better than a Gaussian Model. The accuracy score is 66% which means the program is right 2/3 times. Of course, one way to improve this prediction is by having a greater number than 150 trees but because this data is so large (513,000 rows) we will not do so in order to keep RAM usage lower.

# In[39]:

errors = y_pred - ytest
sns.distplot(errors, bins=5)
plt.title("Errors of Play Type Prediction")
plt.xlim(-2,2);


# This is an error plot of play type prediction. As expected, the peak is at 0 which means for most of the cases the model will accurately predict the play type. The peak at -1 is marginally higher than the peak at 1. This means the model guesses a rush when pass is thrown more than guessing a pass when the ball is ran.

# We can draw 2 conclusions.
# * The variables chosen in the feature matrix are important to determining wether a play will be a pass or run. There may be other variables or another combination of variables needed for better predictions. Other important features may include the matchup and offensive coordinators. The current model treats every game like the same. In actual football games, the gameplan changes every week. Even personnel will play an impact. Teams having star running backs will opt to run the ball more than pass. Then again, if the running back is injured, a team will look to their wide receivers.  Gathering this type of data will assist in better predictions.
# * Also, we must leave some error to the unpredictability in sports. The beauty of sports is that it is often unpredictable. Sometimes, the outcome isn't decided by yards to go, down, point differential...but gut instincts and bold playcalls.
# 

# ## Conclusion

# In conclusion, we learned that yardage gained is simply unpredictable due to so many constantly changing variables. We also learned that the type of play is heavily reliant on the situation before the play. The down, yards to go, time, field position, point differential are important to a coach who's calling the next play. 
# 
# In terms of strategy, through exploring the data we found that pass plays have more variability than rushing plays. Pass plays can gain 0 yards or 20 yards. Rushing plays are more consistent and gain about 3-6 yards every time. Rushing only has the risk of a fumble but passing introduces the risk of an interception and a sack. At the end of it all, teams do pass the ball more than they do run the ball. Only rushing the ball or passing the ball will increase predicitability of the offense making it easier them to defend. Overall, teams must take into account the game situation and gameplan before making bold playcalls in order to win an NFL game.
