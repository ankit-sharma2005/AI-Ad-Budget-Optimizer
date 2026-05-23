import numpy as np # random data generate 
import pandas as pd # to work dataframe data
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import Lasso, Ridge
''' Lasso Regression
weak features ko shrink karta hai
kuch coefficients ko exactly 0 bana deta hai
feature selection karta hai'''
'''Ridge Regression
coefficients ko reduce karta hai
but exactly 0 nahi karta
overfitting reduce karta hai'''
def generate_data_ad():
    np.random.seed(42) # to generate same random number
    n_samples = 200 # generate 200 record
    google_ads = np.random.randint(100,1000,n_samples) # it generate 200 value from 100 to 999
    fb_ads = np.random.randint(100,1000,n_samples)
    # noise data
    newspaper = np.random.randint(10,100,n_samples)
    flyers = np.random.randint(50,100,n_samples)
    radio = np.random.randint(50,200,n_samples)
    # giving weightage to adds
    revenue = (google_ads*2.5)+(fb_ads*1.8)+np.random.normal(0,50,n_samples)
    # np.random.normal(0,50,n_samples) : create noise in data so lasso and rigid work on this dataset . dataset always contain the noisy.
    df = pd.DataFrame({
        'Google_Ads' : google_ads,
        'Facebook_Ads' : fb_ads,
        'Newspaper': newspaper,
        'Flyers': flyers,
        'Radio':radio,
        'Revenue': revenue
    })
    return df
def train_model(df, alpha_val):
    X = df.drop('Revenue',axis = 1)
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    lasso = Lasso(alpha = alpha_val)
    lasso.fit(X_train, y_train)

    ridge = Ridge(alpha = alpha_val)
    ridge.fit(X_train,y_train)
    return lasso, ridge,X_test, y_test, X.columns