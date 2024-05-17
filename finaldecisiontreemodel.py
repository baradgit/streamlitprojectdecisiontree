import base64
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,KFold
from sklearn.tree import DecisionTreeRegressor
import pickle

def create_download_link(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{filename}">Click here to download {filename}</a>'
    return href

st.title("Decision Tree Regressor - Training")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader('Rows and columns')
    st.write(data.shape)

    remove_outliers = st.button('remove_outliers')
  
    if remove_outliers:
       
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.subheader('Removed outliers - rows and columns')
        st.write(data.shape)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    st.write(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    param_grid = {
        'splitter': ['best', 'random'],
        'max_depth': [None, 3,5, 7, 10, 15, 20],
        'min_samples_split': [250, 450, 650, 850, 1000,1250,1500],
        'min_samples_leaf': [200, 400, 600, 800, 1000,1250,1500],  
        'max_leaf_nodes' : [None, 3,5, 7, 10, 15, 20]      
    }

    if st.sidebar.button('Run GridSearchCV'):
        reg = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        st.write(f'Best parameters: {best_params}')

    if st.sidebar.button('Run Algorithm'):
        reg = DecisionTreeRegressor(random_state=42)
        reg.fit(X_train, y_train)
        a = X_train.shape
        st.subheader(f'X_train shape: {a}')
        y_train_pred = reg.predict(X_train)
        mse = mean_squared_error(y_train, y_train_pred)
        st.write(f'Training MSE: {mse}')

        fig, ax = plt.subplots(figsize=(15, 10))
        tree.plot_tree(reg, filled=True, ax=ax, feature_names=X.columns)
        st.pyplot(fig)

        with open('model.pkl', 'wb') as f:
            pickle.dump(reg, f)

        st.success("Model trained and saved successfully!")
        st.markdown(create_download_link('model.pkl'), unsafe_allow_html=True)

        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv('test_data.csv', index=False)
        st.markdown(create_download_link('test_data.csv'), unsafe_allow_html=True)
        st.success("Training Successfully Completed, Now You Can Go Back & Test the Model Using These Downloaded files")
