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

  
    if st.button('remove_outliers'):
       
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
    a = X_train.shape
    st.subheader(f'X_train shape: {a}')
    plt.style.use('fivethirtyeight')

    st.sidebar.markdown("# Decision Tree Regressor")

    splitter = st.sidebar.selectbox(
        'Splitter',
        ('best', 'random')
    )

    max_depth = int(st.sidebar.number_input('Max Depth'))

    min_samples_split = st.sidebar.slider('Min Samples Split',min_value= 1,max_value= X_train.shape[0],step= 2,key=1234)

    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

    max_features = st.sidebar.slider('Max Features', 1, X_train.shape[1], X_train.shape[1],key=1236)

    max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    if st.sidebar.button('Run Algorithm'):
        reg = DecisionTreeRegressor(splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes)
        reg.fit(X_train, y_train)

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
