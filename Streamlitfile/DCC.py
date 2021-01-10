import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import precision_score,recall_score,accuracy_score, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
st.set_option('deprecation.showPyplotGlobalUse', False) # Ignore warnings


def main():
    st.title("Default of Credit Card Clients")
    st.sidebar.title("Default of Credit Card Clients")

    df = pd.read_csv('cleaned_data.csv')

    @st.cache(persist=True)
    def splitting_the_data(df):
        items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
        features_response = df.columns.tolist()
        features_response = [item for item in features_response if item not in items_to_remove]

        X=df[features_response[:-1]]
        y=df['default payment next month']

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=24)
        return X_train,X_test,y_train,y_test


    def plotgraph(curve,model,X_test,y_test):

        if 'Confusion Matrix' in curve:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model ,X_test,y_test,display_labels=['Default','Not Default'])
            st.pyplot()

        if 'ROC Curve' in curve:
            st.subheader("ROC Curve")
            plot_roc_curve(model,X_test,y_test)
            st.pyplot()


        if 'Precision Recall Curve' in curve:
            st.subheader("Precision Recall Curve")
            plot_precision_recall_curve(model,X_test,y_test)
            st.pyplot()


    st.sidebar.subheader('Choose Model')
    model = st.sidebar.selectbox("Model",('Logistic Regression','Random Forest Classifier', 'Decision Tree Classifier'))


    if (model == 'Logistic Regression'):
        st.sidebar.subheader("Choose Hyperparameters")
        C = st.sidebar.number_input("C",0.01,10.0,step=0.01,key='C')
        max_iter = st.sidebar.slider("Maximum Number of Iterations",100,500,key='max_iter')

        curve = st.sidebar.selectbox("Which Curve to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'),key='1')

        if st.sidebar.button("Let's Rocks", key='class'):
            st.subheader("Logistic Regression Result")
            X_train,X_test,y_train,y_test = splitting_the_data(df)
            lr = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', penalty='l2')
            lr.fit(X_train, y_train)

            accuracy = lr.score(X_test,y_test)
            y_pred = lr.predict(X_test)
            st.write("ROC AUC Score:",roc_auc_score(y_test, y_pred).round(2))
            st.write("Accuracy:",accuracy_score(y_test,y_pred).round(2))
            plotgraph(curve,lr,X_test,y_test)


    if (model == 'Decision Tree Classifier'):

        max_depth = st.sidebar.slider("Maximum Depth of Tree",2,5,key='max_depth')

        curve = st.sidebar.selectbox("Which Curve to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'),key='1')

        if st.sidebar.button("Let's Rocks", key='class'):
            st.subheader("Decision Tree Classifier Result")
            X_train,X_test,y_train,y_test = splitting_the_data(df)
            dt = DecisionTreeClassifier(max_depth=max_depth)
            dt.fit(X_train, y_train)
            accuracy = dt.score(X_test,y_test)
            y_pred = dt.predict(X_test)
            st.write("ROC AUC Score:",roc_auc_score(y_test, y_pred).round(2))
            st.write("Accuracy:",accuracy_score(y_test,y_pred).round(2))
            st.write("Precision:",precision_score(y_test,y_pred).round(2))
            st.write("Recall:",recall_score(y_test,y_pred).round(2))
            plotgraph(curve,dt,X_test,y_test)

    if (model == 'Random Forest Classifier'):
        st.sidebar.subheader("Choose Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key='n_est')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,2,step=1,key='max_depth')
        curve = st.sidebar.selectbox("Which Curve to plot?",('ROC Curve','Precision Recall Curve','Confusion Matrix'),key='1')

        if st.sidebar.button("Let's Rocks",key='class'):
            st.subheader("Random Forest Result")
            X_train,X_test,y_train,y_test = splitting_the_data(df)
            rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_test,y_test)
            y_pred = rf.predict(X_test)

            st.write("ROC AUC Score:",roc_auc_score(y_test, y_pred).round(2))
            st.write("Accuracy:",accuracy_score(y_test,y_pred).round(2))
            st.write("Precision:",precision_score(y_test,y_pred).round(2))
            st.write("Recall:",recall_score(y_test,y_pred).round(2))
            plotgraph(curve,rf,X_test,y_test)


if __name__ == '__main__':
    main()
