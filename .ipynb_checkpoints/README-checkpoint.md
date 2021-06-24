# Differentiating Language about Movie Facts from Movie Descriptions using Natural Language Processing 
### Problem Statement
Can we create a machine learning model using natural language processing that can accurately differentiate language about movie facts from movie descriptions?
### Executive Summary
Wouldn’t it be amazing for a computer to be able to identify if you are trying to  tell a fun movie fact or describe a movie? Let’s step through how machine learning can be used to accurately distinguish language used when trying to describe a movie plot and language used when talking about movie facts. 

An easy way to get real language example is from online forums. In this analysis, I used two subreddits r/MovieFinder and r/moviedetails to analyze posts using natural language processing. These posts will act as a guide for the machine to learn how to identify language used when people are trying to describe a movie without using the title between people sharing fun movie facts. With trial and error process of many different models, a logistic regression model is the winner. This model shows great results in accurately differentiating these types of text. Let’s take a deeper look into how this model was discovered.

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**subreddit**|*int64*|full_movie_data_clean|The subreddit the post came from 1 = r/moviefinder, 0 = r/MovieDetails | 
|**title**|*object*|full_movie_data_clean|The selftext and title post combined| 
|**post_length**|*int64*|full_movie_data_clean|The number of words in the title and selftext post combined|
|**subreddit**|*object*|movie_details_df|Name of the subreddit where post is from| 
|**title**|*object*|movie_details_df|The title of the post|
|**selftext**|*object*|movie_details_df|The selftext of the post|
|**subreddit**|*object*|movie_finder_df|Name of the subreddit where post is from| 
|**title**|*object*|movie_finder_df|The title of the post|
|**selftext**|*object*|movie_finder_df|The selftext of the post|


### Conclusions/Recommendations
Logistic Regression with tfidfvectorizer proved to be the best model with an accuracy score of 0.944 on test data. This model had high true positive rate at 0.912, low type I error at 0.024 and low type II Error at 0.088. This model outperformed the 7 other models tested. The model was also able to accurately predict which subeddit a text would belong to with new data not within the model. 

### Limitations
The model was unable to use lemmatized words. It would be interesting in the future to try lemmatized words on a smaller data set to see if the model would improve. I also think it would be beneficial to remove dates when modeling to make the model more universal for larger data or for data in the future that might not talking about 2021, 2019 and 2020 movie dates as frequently. 