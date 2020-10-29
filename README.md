This project allows the user to scrape tweets that contain a specified keyword, and classify tweets as being either positive or negative.
These tweets are then collected in a csv file.
The user must then label some tweets as being either positive or negative, this is because the classification methods used are supervised learning methods.
The python program will then subject the data in the file to sentiment analysis, using support vector machines, naive bayes and 
decision tree classification methods, to learn the pattern from the data and classify new inputs.
The ROC curve and confusion matrix are displayed for each method.

The example dataset provided contains 698 tweets related to the movie "Tenet".
To run the program, ensure you have the following libraries installed:

tweepy, pandas, nltk, scikit-learn, matplotlib, numpy
