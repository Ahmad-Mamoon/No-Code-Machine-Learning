import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def display():
    st.write("\n")
    st.subheader("Displaying Dataset")
    df = st.session_state.df
    st.write(df.shape)
    st.dataframe(df, use_container_width=True)
    st.write("\n")

## TODO: Add option to select multiple columns at once to remove.
def remove_feature():
    df = st.session_state.df
    col_list = df.columns.tolist()
    st.subheader("Select the feature to remove")
    feature = st.selectbox("Features in dataframe are: ", col_list)

    if st.button("Remove", key="remove_button"):
        df.drop(feature, axis=1, inplace=True)
        df.dropduplicates(feature, axis=1, inplace=True)
        st.session_state.df = df
        st.success(f"{feature} removed successfully!")
        # display()

def handle_missing_values():
    df = st.session_state.df
    st.subheader("Handling Missing Values")
    st.write("\n")

    if df.isnull().sum().sum() == 0:
        st.success("No missing values found in the dataset.")
    else:
        st.write("Select the method to handle missing values")
        methods = st.selectbox("Select the method", ("Drop", "Fill with Mean", "Fill with Median", "Fill with Mode"))
        if st.button("Apply", key="missing_value_button"):
            if methods == "Drop":
                df.dropna(inplace=True)
                st.success("Missing values dropped successfully!")
            elif methods == "Fill with Mean":
                df.fillna(df.mean(), inplace=True)
                st.success("Missing values filled with Mean successfully!")
            elif methods == "Fill with Median":
                df.fillna(df.median(), inplace=True)
                st.success("Missing values filled with Median successfully!")
            elif methods == "Fill with Mode":
                df.fillna(df.mode().iloc[0], inplace=True)
                st.success("Missing values filled with Mode successfully!")
            st.session_state.df = df
            # display()

def handle_outliers():
    df = st.session_state.df
    st.subheader("Handling Outliers")
    methods = st.selectbox("Select the method to handle outliers", ("Remove Outliers", "Cap Outliers", "Fill with Median"))
    if st.button("Apply", key="outlier_button"):
        if methods == "Remove Outliers":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            st.success("Outliers removed successfully!")
        elif methods == "Cap Outliers":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR), (Q1 - 1.5 * IQR), df[col])
                    df[col] = np.where(df[col] > (Q3 + 1.5 * IQR), (Q3 + 1.5 * IQR), df[col])
            st.success("Outliers capped successfully!")
        elif methods == "Fill with Median":
            for col in df.columns:
                if df[col].dtype != "object":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.where(df[col] < (Q1 - 1.5 * IQR), df[col].median(), df[col])
                    df[col] = np.where(df[col] > (Q3 + 1.5 * IQR), df[col].median(), df[col])
            st.success("Outliers filled with Median successfully!")
        st.session_state.df = df
        # display()

def categorical_feature_encoding():
    df = st.session_state.df
    st.subheader("Categorical Feature Encoding")
    if len(df.select_dtypes(include="object").columns.tolist()) == 0:
        st.success("No categorical features found in the dataset.")
    else:
        methods = st.selectbox("Select the method to encode categorical features", ("Label Encoding", "One Hot Encoding"))
        if st.button("Apply", key="encoding_button"):
            if methods == "Label Encoding":
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].astype("category").cat.codes
                st.success("Label Encoding applied successfully!")
            elif methods == "One Hot Encoding":
                df = pd.get_dummies(df, drop_first=True)
                st.success("One Hot Encoding applied successfully!")
            st.session_state.df = df
            display()

def split_data():
    df = st.session_state.df
    st.subheader("Splitting the dataset into features and target variable")
    col1, col2 = st.columns(2)
    target = col1.selectbox("Select the target variable", df.columns.tolist())
    sets = col2.selectbox("Select the type of split", ["Train and Test", "Train, Validation and Test"])
    st.session_state.split_type = sets
    col1, col2, col3 = st.columns([1, 0.5, 1])
    if col2.button("Apply", key="split"):
        st.session_state.split_done = True
        if sets == "Train and Test":
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Dataset split successfully. You can now proceed to Building the model.")
        elif sets == "Train, Validation and Test":
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

def model_training():
    st.subheader("Model Selection and Training")
    model_choice = st.selectbox("Select the Model", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
    if st.button("Train Model", key="train_model"):
        if "split_done" in st.session_state and st.session_state.split_done:
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Support Vector Machine":
                model = SVC()
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.success(f"{model_choice} trained successfully!")
        else:
            st.error("Please split the data before training the model.")

def model_evaluation():
    st.subheader("Model Evaluation")
    if "model" in st.session_state:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        st.write(cm)
    else:
        st.error("Please train the model before evaluating.")

def main():
    st.set_page_config(page_title="No Code Machine Learning", page_icon=":computer:", layout="wide")
    st.title("No Code Machine Learning")
    st.subheader("Upload your CSV file")

    with st.spinner("Uploading file..."):
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
        if uploaded_file:
            st.success("File uploaded successfully!")
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            if 'df' not in st.session_state:
                st.session_state.df = df
            display()
            remove_feature()
            handle_missing_values()
            handle_outliers()
            categorical_feature_encoding()
            split_data()
            model_training()
            model_evaluation()



if __name__ == '__main__':
    main()