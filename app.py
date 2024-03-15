import streamlit as st
import pandas as pd
import numpy as np
# import streamlit_scrollable_textbox as stx


def display(df):

    st.write("\n")
    st.subheader("Displaying Dataset")
    st.write(df.shape)
    st.dataframe(df, use_container_width=True)
    st.write("\n")


def remove_feature(df):
    col_list = df.columns.tolist()
    st.subheader("Select the feature to remove")
    feature = st.selectbox("Features in dataframe are: ", col_list)
    if st.button("Remove"):
        df.drop(feature, axis=1, inplace=True)
        st.success(f"{feature} removed successfully!")
        st.write("\n")

    return df

def handle_missing_values(df):
    st.subheader("Handling Missing Values")
    st.write("\n")
    
    if sum(df.isnull().sum().values.tolist()) == 0:
        st.success("No missing values found in the dataset.")

    else:
        st.write("Select the method to handle missing values")
        methods = st.selectbox("Select the method", ("Drop", "Fill with Mean", "Fill with Median", "Fill with Mode"))
        
        if st.button("Apply"):
            if methods == "Drop":
                df.dropna(inplace=True)
                st.success("Missing values dropped successfully!")
                st.write("\n")
            elif methods == "Fill with Mean":
                df.fillna(df.mean(), inplace=True)
                st.success("Missing values filled with Mean successfully!")
                st.write("\n")
            elif methods == "Fill with Median":
                df.fillna(df.median(), inplace=True)
                st.success("Missing values filled with Median successfully!")
                st.write("\n")
            elif methods == "Fill with Mode":
                df.fillna(df.mode().iloc[0], inplace=True)
                st.success("Missing values filled with Mode successfully!")
                st.write("\n")


    return df


def main():
    st.set_page_config(page_title="UI based Machine Learning", page_icon=":computer:", layout="wide")

    st.title("UI based Machine Learning" )
    st.subheader("Upload your CSV file")

    with st.spinner("Uploading file..."):
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

        if uploaded_file:
            st.success("File uploaded successfully!")

            # Read the file
            df = pd.read_csv(uploaded_file)

            # Display the top 5 rows of uploaded file
            dataframe = display(df)
            
            # Remove Feature
            data = remove_feature(df)

            # Handling Missing Values
            data = handle_missing_values(data)

            # Handle Outliers

            # Categorical Feature Encoding

            # Model Selection

            # Number of epochs

if __name__ == '__main__':
    main()