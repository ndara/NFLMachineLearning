
# coding: utf-8

# # Machine Learning

# ### Imports

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os


# ### Let's reload the datasets from our exploration

# In[2]:

pass_df = pd.read_csv('pass_df.csv', index_col=0)
rush_df = pd.read_csv('rush_df.csv', index_col=0)
play_df = pd.read_csv('play_df.csv', index_col=0)


# In[3]:

pass_plays = pd.merge(play_df, pass_df, left_index=True, right_index=True, sort=True)
rush_plays = pd.merge(play_df, rush_df, left_index=True, right_index=True, sort=True)


# Let's also attempt to see if there's a difference using the set of plays with outliers removed. To do this, we will narrow down the datasets to remove outliers.

# In[4]:

rush_plays_no_outliers = rush_plays[((rush_plays.Yards - rush_plays.Yards.mean()) / rush_plays.Yards.std()).abs() < 3]
pass_plays_no_outliers = pass_plays[(((pass_plays.Yards - pass_plays.Yards.mean()) / pass_plays.Yards.std()).abs() < 3) & (pass_plays.Sack == False)]


# To start the machine learning process, we must concatenate the pass plays and rush plays data frames. This allows us to observe every single play and the details of each of the plays. 

# In[5]:

plays = pd.concat([pass_plays, rush_plays]).sort_index()
plays_no_outliers = pd.concat([pass_plays_no_outliers, rush_plays_no_outliers]).sort_index()
plays_no_outliers.head()


# We will also delete the columns that shouldn't have an impact on the questions being observed. These columns will just add meaningless data to the dataframes being observed.

# In[6]:

del plays['INT']
del plays['Direction']
del plays['Drive']
del plays['GameID']
del plays['Location']
del plays['PTS']
del plays['Sack']
del plays['FUM']


# In[7]:

del plays_no_outliers['INT']
del plays_no_outliers['Direction']
del plays_no_outliers['Drive']
del plays_no_outliers['GameID']
del plays_no_outliers['Location']
del plays_no_outliers['PTS']
del plays_no_outliers['Sack']
del plays_no_outliers['FUM']


# In[8]:

plays.head()


# Now we have the parameters that need to be observed and the targets that will be predicted. Can we use the situation before the start of the play to predict how successful the play will be? Let's use Down, YardsToGo, Time, PtDiff, Zone to predict yardage gained. 

# ### Imports for Machine Learning

# We will use sklearn, a machine learning library for Python to train and test our models of prediction.

# In[9]:

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


# ## Predicting Yards Gained

# In[ ]:

X = plays[['Down','PtDiff','Time','YardsToGo','Zone']]


# X is the feature matrix. These are the features that the model will investigate. The features include Down, Point Differential, Time, Yards To Go, Zone of field.

# In[ ]:

y = plays['Yards']


# y is the target array. We want to be able to predict the yardage gained from a play.

# In[ ]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In order to fit the model we muse split the data to train our model. Using that model we can test the remaining data. Sklearn allows us to split up the data.

# ### Use Linear Model to Predict Yardage Gained

# In[ ]:

model = LinearRegression(fit_intercept=False)
model.fit(Xtrain, ytrain)


# In[ ]:

y_pred = model.predict(Xtest)


# In[ ]:

r2_score(y_pred, ytest)


# An r2 value of -30.5 is basically signifying that this data is useless to use when trying to predict yards gained. It is better to guess randomly than to use this model. Let's try to understand why by making an error plot.

# In[ ]:

errors = y_pred - ytest
sns.distplot(errors, bins=30)
plt.title("Errors of Yardage Gain Prediction")
plt.ylabel("Frequency")
plt.xlim(-30,20);


# This is an error plot of the predicted yards gained vs. what was observed. Fortunately, most of the bins are centered around 0. This means that even though the model isn't doing a great job predicting the yardage gained, at least it's getting close. Most predictions seem to be off by +/- 5 yards. The plot is skewed left implying that the predicted values are often overestimating the yardage gained.

# ### Maybe, taking out the outliers and trying to fit a model would be better

# In[ ]:

X = plays_no_outliers[['Down','PtDiff','Time','YardsToGo','Zone']]
y = plays_no_outliers['Yards']


# In[ ]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[ ]:

model = LinearRegression(fit_intercept=False)
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)


# In[ ]:

r2_score(y_pred, ytest)


# This r2 value beats the previous one by a margin of 10. However it is still a negative value, so the model still has very poor performance.

# In[ ]:

errors = y_pred - ytest
sns.distplot(errors, bins=20)
plt.title("Errors of Yardage Gain Prediction")
plt.xlim(-30,20);


# This is another error plot that looks very similar to the first error plot. Again most of the data is centered within +/- 5 yards but this model isn't good enough to pin point how many yards a play will gain.
# 
# Perhaps, a linear model isn't the best choice. Let's repeat the process with a random forest model.

# ### Using a random forest regressor

# In[ ]:

X = plays[['Down','PtDiff','Time','YardsToGo','Zone']]
y = plays['Yards']


# In[ ]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[ ]:

model = RandomForestRegressor(200)
#THIS CODE WILL TAKE A WHILE TO RUN...
model.fit(Xtrain, ytrain)


# In[ ]:

y_pred = model.predict(Xtest)


# In[ ]:

r2_score(y_pred, ytest)


# A random forest regressor is a better model. It increased the r2 value significantly from the linear model. However, it is still negative meaning the model is performing poorly. In order to see this more clearly, let's draw another error plot.

# In[ ]:

errors = y_pred - ytest
sns.distplot(errors, bins=40)
plt.title("Errors of Yardage Gain Prediction")
plt.xlim(-30,20);


# In[ ]:

print("60% confidence interval: " + "( " + str(np.percentile(errors,20)) + ", " + str(np.percentile(errors,80)) + " )" )


# In this plot, more of the data is centered around 0. The model seems to be overestimating the yardage gained from a play. From the confidence interval we see that the model can guess the right amount of yards within 5 yards 60% of the time. This is highly inefficient and these predictions wouldn't be any real good.

# ### Let's try to take out the outliers again

# In[ ]:

X = plays_no_outliers[['Down','PtDiff','Time','YardsToGo','Zone']]
y = plays_no_outliers['Yards']


# In[ ]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[ ]:

model = RandomForestRegressor(200)
#THIS CODE TAKES A WHILE TO RUN
model.fit(Xtrain, ytrain)


# In[ ]:

y_pred = model.predict(Xtest)


# In[ ]:

r2_score(y_pred, ytest)


# An r2 score of -4.5 is marginally better than -4.9. Removing the outliers helped but barely. Unfortuantely, r2 remains a negative value.
# 

# In[ ]:

errors = y_pred - ytest
sns.distplot(errors, bins=20)
plt.title("Errors of Yardage Gain Prediction")
plt.xlim(-30,20);


# This error plot is almost exactly the same as the error plot for random forest regressor with all outliers. This was expected as the r2 values differed by a very slight amount. It does however provide relief to the fact that the model isn't as bad as r2 value says. Much of the data is still centered around 0 +/- 5 yards.

# What we can draw from this is 2 things.
# * The variables chosen in the feature matrix is not a viable predictor of yardage gained. There may be other variables or another combination of variables needed for better predictions. From doing vast amounts of research, better predictions may be possible. In a football game, and in this case multiple football seasons, there are several, several variables that might need to be watched.  For the scope of this project, the chosen variables were not the right ones.
# * The other case is that the yardage gained is simply unpredictable. Big gains in the NFL are usually due to mistakes on the defense or perhaps an extraordinary catch from a wide receiver. These types of events are impossible to predict given every variable in the world. Offensive playcalls and defensive playcalls are often times different for every team, for every quarter, for every matchup. Having enough data accouting for all that is unfeasible.
# 
# Other than coming close to predicting the yardage gained, there isn't much more that can be done to guess the exact yardage gained. Let's now try to predict whether the situation before the play will help to predict whether a team will pass or run.

# ## Predict Type of Play

# In[ ]:

plays.Type = plays.Type.str.replace('RUSH','0')
plays.Type = plays.Type.str.replace('PASS','1')
plays.Type = plays.Type.astype('int64')


# In order to take full advantage of machine learning models, we need to replace the categorical 'Type' of rush and pass and represent them by 0 and 1. 0 will represent a run play. 1 will represent a pass play.

# In[ ]:

plays.head()


# First, let's try using a random forest classifier. Whereas we used random forest regressor for yardage gained, we will use random forest classifier for predicting the type of play. Previously, we measured a continuous variable. Now, we are measuring a categorical variable.

# ### Using a random forest classifier

# In[ ]:

X = plays[['Down','PtDiff','Time','YardsToGo','Zone']]
y = plays['Type']


# In[ ]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[ ]:

#THIS CODE TAKES A WHILE TO RUN....
model = RandomForestClassifier(150)
model.fit(Xtrain, ytrain)


# In[ ]:

y_pred = model.predict(Xtest)


# In[ ]:

accuracy_score(ytest, y_pred)

The random forest classifier returned an accuracy score of 66%. This means that my model will be able to predict the type of play 66% of the time. This is satisfactory and much better than guessing randomly. Overall, 66% is still not as high as desirable. We will follow up with a Gaussian model.
# In[ ]:

errors = y_pred - ytest
sns.distplot(errors, bins=5)
plt.title("Errors of Play Type Prediction")
plt.xlim(-2,2);


# This is an error plot of play type prediction. As expected, the peak is at 0 which means for most of the cases the model will accurately predict the play type. The peak at -1 is marginally higher than the peak at 1. This means the model guesses a rush when pass is thrown more than guessing a pass when the ball is ran.

# ### Using a Gaussian Model

# In[15]:

X = plays[['Down','YardsToGo','PtDiff','Time','Zone']]
y = plays['Type']


# In[16]:

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[17]:

model = GaussianNB()                       
model.fit(Xtrain, ytrain)                  
y_model = model.predict(Xtest)  


# In[18]:

accuracy_score(ytest, y_model)


# Using a gaussian model generates a lower accuracy score than a random forest classifier. By cross validating these models, we see that random forest classifier is a better model but not by a huge margin. Random forest classifier is around 4-5% more accurate than Gaussian.

# We can again draw 2 conclusions.
# * The variables chosen in the feature matrix are important to determining wether a play will be a pass or run. There may be other variables or another combination of variables needed for better predictions. Other important features may include the matchup and offensive coordinators. The current model treats every game like the same. In actual football games, the gameplan changes every week. Even personnel will play an impact. Teams having star running backs will opt to run the ball more than pass. Then again, if the running back is injured, a team will look to their wide receivers.  Gathering this type of data will assist in better predictions.
# * Also, we must leave some error to the unpredictability in sports. The beauty of sports is that it is often unpredictable. Sometimes, the outcome isn't decided by yards to go, down, point differential...but gut instincts and bold playcalls.
# 

# # Challenge Portion

# In[24]:

plays.head()


# In[28]:

plays2 = plays.ix[690000:]
#Truncating data to reduce RAM usage
len(plays2)


# In[16]:

to_cluster = []
for index, row in plays2.iterrows():
    to_cluster.append([row.YardsToGo, row.Yards])


# In[19]:

kmeans = KMeans(n_clusters=2)
kmeans.fit(to_cluster)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


# In[20]:

colors = ['g.','r.']
for i in range(len(to_cluster)):
    plt.plot(to_cluster[i][0], colors[labels[i]])


# In[27]:

plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150)


# From doing a kmeans study, we see that the yards to go and yards gained aren't correlated. If they were correlated the 'X' markers on the graph would be closer together. But from this scatter we see that they are farther away.

# In[ ]:



