
# coding: utf-8

# # Exploration

# In this notebook, I will attempt to make sense of the data that I have previously cleaned. Through a series of charts and proper visualizations, I hope to see specific trends or good predictors of high yardage or touchdowns.

# ## Imports

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os


# ### The previously processed data must be reloaded.

# In[2]:

pass_df = pd.read_csv('pass_df.csv', index_col=0)
rush_df = pd.read_csv('rush_df.csv', index_col=0)
play_df = pd.read_csv('play_df.csv', index_col=0)


# In[3]:

pass_df.head() #Checking to see if CSV is read properly


# In[4]:

rush_df.head() #Checking to see if CSV is read properly


# In[5]:

play_df.head() #Checking to see if CSV is read properly


# ### First, let us examine the outcomes of rushing plays.

# In[6]:

rush_df.head()


# In[7]:

sns.distplot(rush_df.Yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage of all Rushing Plays")
rush_yards = rush_df.Yards
print("mean: " + str(rush_yards.mean()))
print("90% confidence interval: " + "( " + str(np.percentile(rush_yards,5)) + ", " + str(np.percentile(rush_yards,95)) + " )" )


# From this graph, we see that the mean yards gained from a rush play is 4.2 yards. Most running plays fall under the the interval of -2 to 14 yards. From the distplot, we see that the data is heavily centered near the mean.

# Is there any correlation between the direction of the run and the yardage gained?

# In[8]:

RT = rush_df.loc[rush_df.Direction == 'RT'].Yards.mean()
MD = rush_df.loc[rush_df.Direction == 'MD'].Yards.mean()
RE = rush_df.loc[rush_df.Direction == 'RE'].Yards.mean()
LT = rush_df.loc[rush_df.Direction == 'LT'].Yards.mean()
LE = rush_df.loc[rush_df.Direction == 'LE'].Yards.mean()
LG = rush_df.loc[rush_df.Direction == 'LG'].Yards.mean()
RG = rush_df.loc[rush_df.Direction == 'RG'].Yards.mean()
direction_means = [LE,LT,LG,MD,RG,RT,RE]
directions = ["LE","LT","LG","MD","RG","RT","RE"]
sns.barplot(directions,direction_means)
plt.ylabel("Mean Yardage Gained")
plt.xlabel("Direction of Run")
plt.title("Mean Yardage Gained vs. Direction of Run")
temp = rush_df.loc[(rush_df.Direction == 'LE') | (rush_df.Direction == 'LT') | (rush_df.Direction == 'LG') | 
                  (rush_df.Direction == 'MD') | (rush_df.Direction == 'RG') | (rush_df.Direction == 'RT') |
                  (rush_df.Direction == 'RE')]
x = temp.groupby('Direction').mean()
medians = pd.DataFrame(x.Yards).transpose()
medians.columns = ["Left End","Left Guard","Left Tackle","Middle","Right End", "Right Guard", "Right Tackle"]
medians


# From this summary statistic and graph, we see that the least advantegeous direction to run is up the middle. It totals to a mean of 3.87 yards. The best direction to run is left end, which totals to be 5.16 yards. It seems to be a trend that runs farther away from the middle tend to be more successful than those closer to the middle.

# ### Then, let's examine the same distributions for pass plays.

# In[9]:

pass_df.head()


# In[10]:

sns.distplot(pass_df.Yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage of all Passing Plays")
pass_yards = pass_df.Yards
print("mean: " + str(pass_yards.mean()))
print("90% confidence interval: " + "( " + str(np.percentile(pass_yards,5)) + ", " + str(np.percentile(pass_yards,95)) + " )" )


# From this graph, we see that the mean yards gained from a pass play is 6.9 yards. Most pass plays fall under the the interval of 0 to 25 yards. From the distplot, we see that the data has a lot more variablity than the rush distribution. The reason for the extraordinary peak at 0 is the fact this accounts for incomplete passes. Let's observe the distribution for only complete passes.

# In[11]:

comp_yards = pass_df.loc[pass_df.Completion].Yards
sns.distplot(comp_yards)
plt.xlim(-10,40)
plt.ylabel("Frequency")
plt.xlabel("Yards Gained")
plt.title("Distribution of Yardage of all Completions")
print("mean: " + str(comp_yards.mean()))
print("90% confidence interval: " + "( " + str(np.percentile(comp_yards,5)) + ", " + str(np.percentile(comp_yards,95)) + " )" )


# This distribution removes the risk of a pass play being an incompletion. However, it allows us better to visualize how successful a pass is when it is successful. There is a mean of 11.5 yards, which in football is significant. A gain of 10+ yards usually implies a first down. We also see a much wider spread distribution of yards gained ranging from -5 all the way to 40+ yards.

# In[12]:

pass_df.Location.unique()
SL = pass_df.loc[pass_df.Location == 'SL'].Yards.median()
SR = pass_df.loc[pass_df.Location == 'SR'].Yards.median()
SM = pass_df.loc[pass_df.Location == 'SM'].Yards.median()
DL = pass_df.loc[pass_df.Location == 'DL'].Yards.median()
DR = pass_df.loc[pass_df.Location == 'DR'].Yards.median()
DM = pass_df.loc[pass_df.Location == 'DM'].Yards.median()
location_medians = [SL, SM, SR, DL, DM, DR]
locations = ["SL","SM","SR","DL","DM","DR"]
sns.barplot(locations, location_medians)
temp = pass_df.loc[(pass_df.Location == 'SL') | (pass_df.Location == 'SM') | (pass_df.Location == 'SR') | 
                  (pass_df.Location == 'DL') | (pass_df.Location == 'DM') | (pass_df.Location == 'DR')]
m = temp.groupby('Location').median()
medians = pd.DataFrame(m.Yards).transpose()
medians.columns = ["Deep Left","Deep Middle","Deep Right","Short Left","Short Middle", "Short Right"]
medians


# For this analysis, we kept in play all incompletions. That way, the risk of an incompletion when throwing a deep pass vs. a short pass will be accounted for. So as to not allow the data to be skewed by long throws and incompletions, I used median as the summary statistic instead of the mean. We see that more than half the time a deep throw is made, it will be an incompletion. However, short passes are more reliable and show to be good 5 yard gains.

# In[13]:

print("Mean yardage of a pass thrown deep left: " + str(pass_df.loc[(pass_df.Location == 'DL') & (pass_df.Completion)].Yards.median()))
print("Mean yardage of a pass thrown deep right: " + str(pass_df.loc[(pass_df.Location == 'DR') & (pass_df.Completion)].Yards.median()))
print("Mean yardage of a pass thrown deep middle: " + str(pass_df.loc[(pass_df.Location == 'DM') & (pass_df.Completion)].Yards.median()))


# However, we can't quickly draw the conclusion that throwing a deep ball is disadvantegeous because the risk of incompletion is too high. Because given that the pass is completed, chances are the yardage gained from the pass will be close to 25 yards...a quarter of the distance in a football field. Sometimes, risking the long pass may be worth it.

# ### Run vs. Pass
# We see that running the ball gives us consistent averages of 4-5 yards. However, passing the ball short will give us up to 5-6 yards at the small risk of incompletion. Throwing a deep ball will generate close to 24 yards at the expense of high risk of incompletion. How do NFL teams decide what play type to use? Given circumstances of the game, we can observe more closely the real practices of NFL plays. Doing so will help us predict the yard gain of the next play.

# ### Now that we have a grasp of rushing and passing plays, let's observe the conditions of the game into play.

# In[14]:

play_df.head()


# In[15]:

pass_plays = pd.merge(play_df, pass_df, left_index=True, right_index=True, sort=True)
pass_plays.head()


# In[16]:

rush_plays = pd.merge(play_df, rush_df, left_index=True, right_index=True, sort=True)
rush_plays.head()


# In[17]:

assert (((len(rush_plays) + len(pass_plays))) < len(play_df))


# In[18]:

rush_plays_no_outliers = rush_plays[((rush_plays.Yards - rush_plays.Yards.mean()) / rush_plays.Yards.std()).abs() < 3]
sns.boxplot(rush_plays_no_outliers.Down, rush_plays_no_outliers.Yards)
plt.title("Rushing Plays - Yards Gained vs. Down")
rush_plays.Down.value_counts(normalize=True)


# This is a graph that depicts the distribution of yardage gained for running plays split up by the down it was run. Looking at the graph, it looks like the best time to run is 3rd down as the median and 3rd quartile are above other downs. 4th down is a risky time to run unless the team only has a few yards to go. Upon looking at the value counts, we get a better idea of what happens during NFL games. More than half the time the ball is run, it is on first down. Most 4th downs are usually punts and field goals, so only 1.3% of the time is the ball ran on 4th down. 

# In[19]:

pass_plays_no_outliers = pass_plays[(((pass_plays.Yards - pass_plays.Yards.mean()) / pass_plays.Yards.std()).abs() < 3) & (pass_plays.Sack == False)]
sns.boxplot(pass_plays_no_outliers.Down, pass_plays_no_outliers.Yards)
plt.title("Passing Plays - Yards Gained vs. Down")
rush_plays.Down.value_counts(normalize=True)


# Unlike running plays, passing plays have more variability. It seems to be that many of the short plays are due to screen passes to a wide receiver or running back. For this boxplot, I only considered pass plays that didn't end in a sack. Surprisingly, the variability of passing plays is essentially the same for every down.

# In[20]:

play_df.groupby('Down').Type.value_counts(normalize=True)


# This is an important statistic because it shows the proportion of Rush and Pass plays for each down. For first and second down, it is a tossup as to whether the ball will be passed or thrown. On third down, teams elect to pass the ball 3 out of 4 times. When a team goes for it on 4th down, chances are the team will pass the ball.

# In[50]:

#play_df[(play_df.PtDiff <= 14) & (play_df.PtDiff >= -14)].groupby('PtDiff').Type.value_counts(normalize=True)
temp = play_df[(play_df.PtDiff <= 14) & (play_df.PtDiff >= -14)]
plt.figure(figsize=(40,20))
sns.countplot(x="PtDiff", hue="Type", data=temp)


# By simple observation of this data, we see that when teams are losing, they tend to pass the ball more. When a team is winning, they tend to rush the ball more. Rushing the ball, although disadvantageous when trying to gain yards, is safer than passing risking an incompletion or worse, an interception.

# In[51]:

#play_df[(play_df.YardsToGo <= 20)].groupby('YardsToGo').Type.value_counts(normalize=True)
temp = play_df[(play_df.YardsToGo <= 20)]
plt.figure(figsize=(30,10))
sns.countplot(x="YardsToGo", hue="Type", data=temp)


# From these value count proportions, we see that teams that have fewer yards to go for the first down will run the ball. Teams that have more yards to go will throw the ball. This was expected because passing will generally return more yards than a rush play.

# In[ ]:



