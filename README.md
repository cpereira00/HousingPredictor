# HousingPredictor
Housing Predictor is a Machine Learning project that aims to predict housing prices in San Antonio, Texas through the use of multivariate linear regression.

## I. Data Gathering
I gathered and extracted data by scraping multiple web pages from a realtor site through the use of the beautifulsoup library. I then cleaned the dataset and extrapolated 3 features from each house; specifically the # of bedrooms, bathrooms and the size of the house. *Disclaimer: The use of the data is for educational purposes only.*

## II. Training
Using scikit-learn, the data gathered was divided into a 70/30 train/test split. I then standardized the feature vectors as it was clear the number of bedrooms/bathrooms weren't on the same scale as the size of the house and would create a bias.  

## III. Testing and Plotting
