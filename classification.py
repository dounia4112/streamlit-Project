import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def app():
    st.title('✅ Classification')

    if 'data' in st.session_state:
        df = st.session_state['data']
        st.write('Data loaded for classification')
        st.write(df.head())

        # -----------------------------
        
        target_column = df.columns[-1] 
        st.write('Target Column ', target_column)

        # -----------------------------
        X = df.drop(columns=target_column)
        y = df[target_column]
        # -----------------------------

        test_size = st.slider("Test Size (as %)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # -----------------------------
        algorithm = st.selectbox('Choose Algorithm', ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbor'])

        if algorithm == 'Logistic Regression':
            clf = LogisticRegression()

        elif algorithm == 'Decision Tree':
            criterion = st.radio('Type of Criterion', ['gini', 'entropy'])
            max_depth = st.slider('Max Depth', 5, 20,5)
            min_sample_leaf = st.slider('Minimum Sample Leaf', 10, 50,10)
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_sample_leaf)

        elif algorithm == 'Random Forest':
            n_estimators = st.slider('Number of Estimators', 10, 50,10)
            max_depth = st.slider('Max Depth', 5, 20,5)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        else:
            n_neighbors = st.slider('Number of Neighbors', 10, 50,10)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)

        # -----------------------------

        if st.button('Train Model'):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            tab1, tab2 = st.tabs(['Metrics', 'Confusion Matrix'])
            with tab1:
                st.write('### Performance Metrics')
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with tab2:
                st.write('### Confusion Matrix')
                cm = confusion_matrix(y_test, y_pred)
                fig_corr = px.imshow(cm, text_auto=True)
                st.plotly_chart(fig_corr)

    else:
        st.write('No data found. Please upload a file on the EDA page first.')