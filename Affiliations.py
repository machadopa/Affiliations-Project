import pandas as pd
import sklearn.naive_bayes
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import jarowinkler as jw

def read_data(path:str)->pd.DataFrame:
    """

    :param path:
    :return:
    """

    df = pd.read_csv(path)

    affiliations = df['"name"'].values

    out = []
    for i,aff1 in enumerate(affiliations):
        for j, aff2 in enumerate(affiliations[i+1:]):
            out.append(((aff1,aff2),(i,j)))

    df = pd.DataFrame(out, columns=['name','indices'])
    jwscore = lambda x: jw.jarowinkler_similarity(x[0],x[1])
    df['jwdist'] = df['name'].apply(jwscore)  # New column that gives the similarity score
    return df

def split_data(df:pd.DataFrame):
    """
    Splits the data frame by features and labels
    :param df: pandas dataframe
    :return: features and labels train and test sets
    """
    X = df[df.columns[~df.columns.isin(['jwdist'])]]  # Creating a data frame with only the features
    y = df[['match']]  # Creating a data frame with only the label(What I'm trying to predict)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
    return X_train,X_test,y_test,y_train

def trainer(X_train:pd.DataFrame,y_train:pd.DataFrame):
    """
    Practice for using set features to predict the data
    :param X_train: Features that will be used to train the model
    :param y_train: Labels that will be used to train the model
    :return: model
    """
    model = sklearn.naive_bayes.GaussianNB()
    model.fit(X_train,y_train.values.ravel(),sample_weight=None)
    return model

def possess(model,X_test:pd.DataFrame,Y_test:pd.DataFrame):
    '''

    :param model: Model
    :param X_test: features that will be tested
    :param Y_test: Labels that will be tested
    :return:
    '''
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Compare for accuracy
    accuracy = accuracy_score(Y_test, predictions)
    return accuracy

def predict(features: pd.DataFrame) -> np.ndarray:

    features = features[features.columns]
    return features

if __name__ == "__main__":

    affiliations = read_data("C:\\Users\\FM Inventario\\Documents\\affiliations.csv")
    print(affiliations)

    split_data(affiliations)

    training_set = ['llc1ylm7JG',]

