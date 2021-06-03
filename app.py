import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

# Import Estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score

# Import metric evaluators
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Split the data into features and label
X = df.drop('Outcome', axis=1)
y = df['Outcome']
    
# Split X,y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0, shuffle=True)

def check_model(model):
    if model == 'Random Forest Classifier':
        model = RandomForestClassifier()
    elif model == 'KNN':
        model = KNeighborsClassifier()
    else:
        model = SVC()
    return model

def set_params(model):
    """
    Performs parameter tuning on the selected model
    model: the selected estimator
    """
    params = {}

    if model == 'Random Forest Classifier':
        n = st.sidebar.slider('N Estimator',10, 1200)
        max_depth = st.sidebar.slider('Max Depth', 1, 50)
        leaf = st.sidebar.slider('Min Samples Leaf', 1,50)
        split = st.sidebar.number_input('Min Samples split', 2,20)

        params['n_estimators'] = int(n)
        params['max_depth'] = max_depth
        params['min_samples_leaf'] = leaf
        params['min_samples_split'] = split

    elif model == 'KNN':
        n_neighbors = st.sidebar.slider('N Neigbors', 1,100)
        
        params['n_neighbors'] = n_neighbors
    else:
        c = st.sidebar.slider('C',0.01,  10.0)
        gamma =st.sidebar.slider('Gamma', 1,10)
        degree = st.sidebar.slider('Degree',1,10)
        kernel = st.sidebar.selectbox('Kernel',['RBF'])

        params['C'] = c
        params['gamma'] = gamma
        params['degree'] = degree
        params['kernel'] = kernel.lower()

    return params

# Set model Hyper Parameters
def add_params_to_model(model, params):
    
    if model == 'Random Forest Classifier':
        model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'])
    elif model == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    else:
        model = SVC(kernel=params['kernel'], C = params['C'], gamma=params['gamma'], degree=params['degree'])
    return model

# Score the model
def score_model(df, model):
    """
    df: data frame
    model: estimator chosen for training
    """
    try:
        np.random.seed(0)
        params = set_params(model)
        model = add_params_to_model(model, params)
        model.fit(X_train, y_train)

        y_preds = model.predict(X_test)

        score = model.score(X_test, y_test)
        accuracy = accuracy_score(y_test, y_preds)
        cross_val = cross_val_score(model, X, y, cv=5)
        conf_matrix = confusion_matrix(y_test, y_preds)
        report = classification_report(y_test, y_preds)

        evaluations = {
            'Score': score,
            'Accuracy': accuracy,
            'Cross Validation Score': cross_val,
            'Classification Report': report,
            'Confusion Matrix': conf_matrix
        }
        return evaluations
    except Exception as e:
        return st.error(e)


def run():
    st.balloons()
    st.title('Diabetes Prediction Model Trainer')
    model = st.sidebar.selectbox('Select Model',['Random Forest Classifier', 'KNN', 'LinearSVC', 'Logistic Regression'])
    st.write('''### Score''')
    sc_model = score_model(df, model)
    for key, val in sc_model.items():
        st.write(f'### {key}')
        st.success(val)

    st.write(sc_model)
if __name__ == '__main__':
    st.cache(suppress_st_warning=True)
    run()