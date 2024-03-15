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


def handle_outliers(df):
    st.subheader("Handling Outliers")
    st.write("\n")

    st.write("Select the method to handle outliers")
    methods = st.selectbox("Select the method", ("Remove Outliers", "Cap Outliers", "Fill with Median"))

    if st.button("Apply"):
        if methods == "Remove Outliers":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            st.success("Outliers removed successfully!")
            st.write("\n")

        elif methods == "Cap Outliers":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR), (Q1 - 1.5 * IQR), df[col])
                    df[col] = np.where(df[col] > (Q3 + 1.5 * IQR), (Q3 + 1.5 * IQR), df[col])
            st.success("Outliers capped successfully!")
            st.write("\n")

        elif methods == "Fill with Median":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR), df[col].median(), df[col])
                    df[col] = np.where(df[col] > (Q3 + 1.5 * IQR), df[col].median(), df[col])
            st.success("Outliers filled with Median successfully!")
            st.write("\n")

    return df


def categorical_feature_encoding(df):
    st.subheader("Categorical Feature Encoding")
    st.write("\n")

    if len(df.select_dtypes(include="object").columns.tolist()) == 0:
        st.success("No categorical features found in the dataset.")
    else:
        st.write("Select the method to encode categorical features")
        methods = st.selectbox("Select the method", ("Label Encoding", "One Hot Encoding"))

        if st.button("Apply"):
            if methods == "Label Encoding":
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].astype("category").cat.codes
                st.success("Label Encoding applied successfully!")
                st.write("\n")

            elif methods == "One Hot Encoding":
                df = pd.get_dummies(df, drop_first=True)
                st.success("One Hot Encoding applied successfully!")
                st.write("\n")

    return df


def split_data(df):
    st.subheader("Splitting the dataset into features and target variable")
    st.write("\n")

    col1, col2 = st.columns(2)
    target = col1.selectbox("Select the target variable", df.columns.tolist())
    sets = col2.selectbox("Select the type of split", ["Train and Test", "Train, Validation and Test"])
    st.session_state.split_type = sets
    col1, col2, col3 = st.columns([1, 0.5, 1])
    if col2.button("Apply", key="split"):
        st.session_state.split_done = True
    
        if sets == "Train and Test":
            from sklearn.model_selection import train_test_split
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Dataset split successfully. You can now proceed to Building the model.")

        elif sets == "Train, Validation and Test":
            from sklearn.model_selection import train_test_split
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)
            X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.X_val = X_val
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_val = y_val
            st.success("Dataset split successfully. You can now proceed to Building the model.")

    return X, y

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
            data = handle_outliers(data)

            # Categorical Feature Encoding
            data = categorical_feature_encoding(data)

            # Splitting the dataset into features and target variable
            X, y = split_data(data)


            # Model Selection

            
            # Number of epochs

if __name__ == '__main__':
    main()