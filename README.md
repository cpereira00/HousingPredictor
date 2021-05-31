# HousingPredictor
Housing Predictor is a Machine Learning project that aims to predict housing prices in San Antonio, Texas, through the use of multivariate linear regression.

## I. Data Gathering
I gathered and extracted data by scraping multiple web pages from a realtor site through the use of the beautifulsoup library. I then cleaned the dataset and extrapolated 3 features from each house; specifically the # of bedrooms, bathrooms and the size of the house. *Disclaimer: The use of the data is for educational purposes only.*

## II. Training
Using the scikit-learn library, the data gathered was divided into a 70/30 train/test split. I then standardized the feature vectors as it was clear the number of bedrooms/bathrooms weren't on the same scale as the size of the house and would create a bias. I then used the training set to train the multivariate linear regression model and used the test set to predict the housing prices.  

## III. Testing and Plotting
Finally, through the use of the matplotlib library, I was able to plot the individual features against the housing prices as well as the plot of the prediction model against the test data set. For testing, I then used the mean square error to find the squared difference between the test values and predicted values of the pricing of the houses. Furthermore, I determed the Root mean square error and as well as the coefficient of determination (the score) for my testing.
