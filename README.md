# Movie-Recommendation-System

This Movie Recommendation System uses the movielens dataset with over 1,000,000 movie ratings and 80,000 movies.

The model starts off by loading the dataset and creating vectors to represent each movie based on the relevance of their tags/themes.

After creating vectors to represent the movies, the model creates vectors representing the dataset-users movie preferences of tags/themes (based on the movie vectors and the ratings given by the users).

The model takes input for movies the user has previously watched and creates a vector to represent the user's preferences.

When the user requests recommendations, the model finds the dataset-users with the most similar movie tastes using the k-nearest neighbors algorithm and returns the most popular movies by the similar movies, this is a process known as collaborative filtering.

This model was coded in Python and used the following libraries: pandas; numpy; scikit-learn; ipywidgets
