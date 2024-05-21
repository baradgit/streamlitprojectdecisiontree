import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.model_selection import GridSearchCV, KFold

st.title("Decision Tree Regressor")

# Upload the dataset
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader('Rows and columns')
    st.write(data.shape)

    # Handle NaN values
    nan_handling = st.sidebar.selectbox(
        "How would you like to handle NaN values?",
        ("Drop rows", "Fill with mean", "Fill with median", "Fill with mode")
    )

    if nan_handling == "Drop rows":
        data = data.dropna()
    elif nan_handling == "Fill with mean":
        data = data.fillna(data.mean())
    elif nan_handling == "Fill with median":
        data = data.fillna(data.median())
    elif nan_handling == "Fill with mode":
        data = data.fillna(data.mode().iloc[0])

    st.subheader('Dataset after handling NaN values')
    st.write(data.shape)

    # Button to remove outliers
    if st.button('remove_outliers'):
        # Remove outliers using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.subheader('Removed outliers - rows and columns')
        st.write(data.shape)

    # Ensure the data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Check for NaN values after conversion and handle them if any
    if data.isnull().sum().sum() > 0:
        st.warning("Data contains NaN values after conversion. Please clean your data.")
    else:
        # Assume that the last column is the target variable and the rest are features
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

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

        max_depth = st.sidebar.number_input('Max Depth', min_value=0, step=1, value=0)

        min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=X_train.shape[0], step=1, key=1234)

        min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=1235)

        max_features = st.sidebar.slider('Max Features', 1, X_train.shape[1], X_train.shape[1], key=1236)

        max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes', min_value=0, step=1, value=0)

        if st.button('checkgridsearch'):
            regs = DecisionTreeRegressor()

            # Define the parameters for GridSearchCV
            param_grid = {
                'splitter': ['best', 'random'],
                'max_depth': [None, 3, 5, 7, 10, 15, 20],
                'min_samples_split': [2, 10, 50, 100, 200],
                'min_samples_leaf': [1, 2, 5, 10, 20],
                'max_leaf_nodes': [None, 5, 10, 20, 30]
            }

            # Create a GridSearchCV object
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            grid_search = GridSearchCV(regs, param_grid, cv=cv)
            st.subheader('Grid Search')
            st.write(X.shape, y.shape)
            # Fit the model to the training data
            try:
                grid_search.fit(X_train, y_train)
                # Print the best parameters
                strs = grid_search.best_params_
                st.subheader(strs)
            except ValueError as e:
                st.error(f"Grid search failed: {e}")

        if st.sidebar.button('Run Algorithm'):

            if max_depth == 0:
                max_depth = None

            if max_leaf_nodes == 0:
                max_leaf_nodes = None

            try:
                reg = DecisionTreeRegressor(
                    splitter=splitter,
                    max_depth=max_depth,
                    random_state=42,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes
                )
                reg.fit(X_train, y_train)

                # Predict on training data
                y_train_pred = reg.predict(X_train)
                # Predict on testing data
                y_test_pred = reg.predict(X_test)

                # Calculate Mean Squared Error on training data
                mse_train = mean_squared_error(y_train, y_train_pred)
                # Calculate Mean Squared Error on testing data
                mse_test = mean_squared_error(y_test, y_test_pred)

                st.subheader(f"Mean Squared Error for Decision Tree Regressor on Training Data: {mse_train:.2f}")
                st.subheader(f"Mean Squared Error for Decision Tree Regressor on Testing Data: {mse_test:.2f}")

                # Export the decision tree to a tree structure
                tree_data = export_graphviz(reg, feature_names=data.columns[:-1])

                # Display the tree structure using Streamlit's graphviz_chart
                st.graphviz_chart(tree_data)
            except ValueError as e:
                st.error(f"Model fitting failed: {e}")
