import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def app():
    st.title('📊 Exploratory Data Analysis (EDA)')

    uploader = st.file_uploader('Upload a CSV file')

    if uploader:
        df = pd.read_csv(uploader)
        st.session_state['data'] = df 
        # -----------------------------
        st.write('### Dataset Preview')
        st.write(df.head())

        # -----------------------------
        st.write('### Descriptive Statistics')
        st.write(df.describe())

        # -----------------------------
        st.write('### Correlation Matrix')
        if st.checkbox('Show Correlation Matrix'):
            corr_matrix = df.corr(numeric_only=True)
            fig_corr = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig_corr)

        # -----------------------------
        st.write('### Advanced Visualizations')
        st.write('Select Column for Visualization')

        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) >= 2:
            x_axis = st.selectbox('X-Axis', num_cols)
            y_axis = st.selectbox('Y-Axis', [col for col in num_cols if col != x_axis])
            fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title='Scatter Plot')
            st.plotly_chart(fig_scatter)
        else:
            st.warning('Not Enough Numerical Columns')

        # -----------------------------
        st.write('### Data Distribution')
        column = st.selectbox('Features', num_cols)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df[column], color='blue', edgecolor='black', bins=20)
        ax_hist.set_title(f"Histogram of {column}")
        ax_hist.set_xlabel(column)
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)

        # -----------------------------

