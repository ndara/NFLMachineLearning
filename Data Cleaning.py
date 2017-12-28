
# coding: utf-8

# # Data Cleaning

# The Armchair Analysis dataset is a gem to do data science with. Unlike most datasets data scientists work with, this one is remarkably clean and formatted. Thus, very little cleaning is necessary. Mostly, it will be dealing with stripping off columns of a table that will not be used in the analysis.

# In[28]:

prefix = '/data/armchair-nfl/csv/' #This is where the data is stored


# ## Imports

# In[29]:

import pandas as pd
import os


# ### First, let us clean up the pass csv such that we can see how successful a pass play is.

# In[30]:

pass_df = pd.read_csv(prefix + 'PASS.csv', sep = ',', index_col=0)
pass_df.head()


# In[31]:

del pass_df['psr']
del pass_df['trg']
del pass_df['succ']
del pass_df['spk']
del pass_df['dfb']


# The data of importance is whether a pass is completed, and the yardage gained by the pass. The other columns are not needed.

# In[32]:

pass_df.columns = ['Location','Yards','Completion']
pass_df.Yards = pass_df.Yards.astype('int64')
pass_df.Completion = pass_df.Completion.astype('bool')
pass_df.Location = pass_df.Location.astype('category')


# In[33]:

pass_df.info()
pass_df.head()


# In[34]:

assert list(pass_df.columns)==['Location', 'Yards', 'Completion']
assert pass_df.Yards.dtype.name=='int64'
assert pass_df.Completion.dtype.name=='bool'
assert pass_df.Location.dtype.name=='category'


# The assert tests are validation that the pass csv has been polished and ready for exploration.

# ### Now, let us clean up the rush csv to see how successful run plays are.

# In[35]:

rush_df = pd.read_csv(prefix + 'RUSH.csv', sep = ',', index_col=0)
rush_df.head()


# In[36]:

del rush_df['bc']
del rush_df['succ']
rush_df = rush_df[rush_df.kne == 0] 
#We don't want to analyze kneel downs as no effort to run was made. Although technically, it is a run play,
#it is impractical to involve in an analysis.
del rush_df['kne']


# The data of importance is which direction the rush went, and how much yardage was gained. For this analysis, the identity of ball carrier will be ignored.

# In[37]:

rush_df.columns = ['Direction','Yards']
rush_df.Direction = rush_df.Direction.astype('category')
rush_df.Yards = rush_df.Yards.astype('int64')


# In[38]:

assert list(rush_df.columns)==['Direction','Yards']
assert rush_df.Yards.dtype.name=='int64'
assert rush_df.Direction.dtype.name=='category'


# The assert tests are validation that the rush csv has been polished and ready for modeling.

# ### We will now clean up the play csv

# In[39]:

play_df = pd.read_csv(prefix + 'PLAY.csv', sep = ',', index_col=1)
play_df.head()


# In[40]:

del play_df['off']
del play_df['def']
play_df = play_df[(play_df['type'] == 'RUSH') | (play_df['type'] == 'PASS')] #We are not interested in other types of plays.
# To this analysis only pass plays and rush plays matter. We will be ignoring kickoffs, punts, extra points, and penalties.
del play_df['len']
del play_df['sec'] #The time on the game clock is important but the seconds is too specific and unnecessary.
del play_df['timo']
del play_df['timd']
del play_df['yfog']
del play_df['fd']
del play_df['sg']
del play_df['nh']
del play_df['tck']
del play_df['pen']
del play_df['saf']
del play_df['blk']
del play_df['olid']



# In order to keep the data clean, I have deleted many of the columns that will be unnecessary to the analysis.

# In[41]:

play_df['Point Diff'] = play_df.ptso - play_df.ptsd
del play_df['ptso']
del play_df['ptsd']


# I have deleted the scoreboard statistics and replaced it with a point differential column. This reduces the columns in my data table while not losing significance.

# In[42]:

play_df['Time'] = play_df.qtr*15 - play_df['min']
del play_df['qtr']
del play_df['min']
play_df.head()


# I have also deleted both the quarter and minute columns. I have replaced it with a time (in minutes) column, this will save the number of columns to have in the dataframe with no loss in information.

# In[1]:

play_df.columns = ['GameID','Type','Drive','Down','YardsToGo','Zone','PTS','Sack','INT','FUM','PtDiff','Time']
play_df.Type = play_df.Type.astype('category')
play_df.Time = play_df.Time.astype('int64')
play_df.Down = play_df.Down.astype('category')
play_df.Zone = play_df.Zone.astype('category')
play_df.PTS = play_df.PTS.astype('category') 
play_df.Sack = play_df.PTS.astype('bool')
play_df.INT = play_df.INT.astype('bool')
play_df.FUM = play_df.FUM.astype('bool')
play_df.info()
play_df.head()


# In[44]:

assert list(play_df.columns)==['GameID','Type','Drive','Down','YardsToGo','Zone','PTS','Sack','INT','FUM','PtDiff','Time']
assert play_df.GameID.dtype.name=='int64'
assert play_df.Type.dtype.name=='category'
assert play_df.Drive.dtype.name=='int64'
assert play_df.Time.dtype.name=='int64'
assert play_df.Down.dtype.name=='category'
assert play_df.YardsToGo.dtype.name=='int64'
assert play_df.Zone.dtype.name=='category'
assert play_df.Zone.dtype.name=='category'
assert play_df.Sack.dtype.name=='bool'
assert play_df.INT.dtype.name=='bool'
assert play_df.FUM.dtype.name=='bool'
assert play_df.PtDiff.dtype.name=='int64'


# The assert tests are validation that the play csv has been polished and ready for modeling.

# ## To save these results, we will write these dataframes to files.

# In[45]:

play_df.to_csv("play_df.csv")
rush_df.to_csv("rush_df.csv")
pass_df.to_csv("pass_df.csv")


# This concludes the data cleaning process of data science. I have successfully narrowed down loads of tables and columns to the specific fields I want to analyze. Next, in the exploration notebook, I will represent the different trends present in the observed data.

# In[ ]:



