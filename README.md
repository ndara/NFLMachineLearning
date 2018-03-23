Experimentation of application of data science to NFL 2015 dataset to predict the success of any given play.
# NFL Analysis of Individual Plays

> To be a successful team in the NFL, you must have a combination of talent and strategy. Being able to predict outcomes of games and player performances rests heavily on being able to understand a play by play analysis. With this in mind, the following notebooks will illustrate important relationships of play result depending on different aspects of the game. Through an analysis of statistics and facts from the last 15 years of NFL football, the intended result is to be able to decide what the outcome of a play will be given initial conditions. This includes what type of play it will be and the yards gained.

### Armchair Analysis Dataset

> The dataset that will be used for the project is 'Armchair Analysis'. This is a bundle of CSV files that hold stats for essentially every single type of play, player, and team in the game for the last 15 years. Obviously this dataset is quite thorough and it would be impractical to analyze every file. For my project, I will be picking out a few CSV's to achieve the goal of predicting whether a play will be successful and how successful it will be. Specifically, I will be using the PLAY, RUSH, PASS files.

### Questions of Consideration

* Is passing or running the ball more advantegeous?
* Does the advantage depend on what down it is?
* Do teams change gameplans depending on point differntial?
 
 Ultimately the crux of this data analysis focuses on these 2 questions.
* Can the yardage gained from an upcoming play be predicted by knowing the game clock, down, point differential, field position, and yards to go?
* Using the same features, perhaps, can we predict whether a team will opt to pass or rush the ball?
 

### Descriptions of individual notebooks

`Data Cleaning.py`: In the data cleaning notebook, I have narrowed down the csv files important to my project. From those csv's I have converted them to Pandas dataframes and deleted columns that are insignificant. Then, I have changed the columns to appropriate data types. I also simplified multiple columns to single columns to save complexity of the dataframe. There are assert tests to ensure that the data has been sufficiently and properly cleaned.

`Exploration.py`: In the exploration dataset, I let my data loose to see what trends and interesting patterns may emerge. I explored the differences of rushing and passing the ball. I grouped my data by which down the play was conducted and repeated to observe correlations. I also managed to explore the effect point differential has on teams along with yards to go.

`Machine Learning.py`: For the machine learning aspect of the project, I used several models to create predictions. To predict yardage gained, I used a linear regression, and random forest regressor. I also removed outliers to try to achieve higher precision. To predict the type of play, I used a gaussian model and cross validated it with a random forest classifier. 

`Presentation.py`: This notebook condenses the data science project in a presentable format. Taking excerpts from the other notebooks, the presentation creates a flowing narrative that observes the questions addressed in detail. 
