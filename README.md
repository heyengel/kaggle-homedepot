# Home Depot Product Search Relevance Challenge

Current score: 0.48619 (RMSE)

This is a challenge to predict the search relevance of search results on homedepot.com. More than 73% of the products in the dataset were unique items, which presented a challenge in training the model.

I used natural language processing (NLTK SnowballStemmer) to derive the word stems on the product title, description and search terms.  I then created features based on word ratios between the search terms and the product title and description.

Random Forest models and parameter grid search was used to generate the relevance predictions.
