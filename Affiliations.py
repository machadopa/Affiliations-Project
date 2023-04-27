import jarowinkler
import pandas as pd
import sklearn.naive_bayes
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import jarowinkler as jw

def read_data(path:str)->pd.DataFrame:
    """
    Read dataset from working directory
    Adjust dataframe to specific format.

    :param path:String containing path name
    :return: new dataframe
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
    model = xgb.XGBClassifier()
    model.fit(X_train,y_train)
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

def predict_model(features: pd.DataFrame,model) -> np.ndarray:

    prediction = model.predict(features)
    return prediction

if __name__ == "__main__":

    affiliations = read_data("C:\\Users\\FM Inventario\\Documents\\affiliations.csv")
    print(affiliations)

    jwscore = lambda x: jw.jarowinkler_similarity(x[0], x[1])

    training_set = [[jarowinkler.jarowinkler_similarity('llc1ylm7JG','llc1ylo7JG'),0],
                    [jarowinkler.jarowinkler_similarity('hello','hello'),1],
                    [jarowinkler.jarowinkler_similarity('glass','owl'),0],
                    [jarowinkler.jarowinkler_similarity('clown','blown'),0],
                    [jarowinkler.jarowinkler_similarity('eathdkewederdknvuewoi','eathdkewederdknvuewoi'),1]]
    #dataframe conversion
    train = pd.DataFrame(training_set,columns=['jwdist','match'])

    train_x = train['jwdist']
    train_y = train['match']

    aff_model = trainer(train_x,train_y)

    affiliations['match'] = predict_model(affiliations['jwdist'],aff_model)

    print(affiliations)





