import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def display():
    st.write("\n")
    st.subheader("Displaying Dataset")
    df = st.session_state.df
    st.write(df.shape)
    st.dataframe(df, use_container_width=True)
    st.write("\n")


import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df):
    st.subheader("Histogram")
    column = st.selectbox("Select column for histogram", df.columns)
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True)
    st.pyplot(plt.gcf())

def plot_scatter(df):
    st.subheader("Scatter Plot")
    columns = st.multiselect("Select two columns for scatter plot", df.columns)
    if len(columns) == 2:
        plt.figure(figsize=(10, 4))
        sns.scatterplot(data=df, x=columns[0], y=columns[1])
        st.pyplot(plt.gcf())

def plot_box(df):
    st.subheader("Box Plot")
    numerical_df = df.select_dtypes(include=['number'])
    column = st.selectbox("Select column for box plot", numerical_df.columns)
    plt.figure(figsize=(10, 4))
    sns.boxplot(y=df[column])
    st.pyplot(plt.gcf())

def plot_heatmap(df):
    st.subheader("Correlation Heatmap")
    numerical_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())


def display_graphs():
    df = st.session_state.df
    st.subheader("Data Visualization")
    st.write("Select the types of graphs to display:")
    
    plot_histogram_checkbox = st.checkbox("Histogram")
    plot_scatter_checkbox = st.checkbox("Scatter Plot")
    plot_box_checkbox = st.checkbox("Box Plot")
    plot_heatmap_checkbox = st.checkbox("Correlation Heatmap")

    if plot_histogram_checkbox:
        plot_histogram(df)
    if plot_scatter_checkbox:
        plot_scatter(df)
    if plot_box_checkbox:
        plot_box(df)
    if plot_heatmap_checkbox:
        plot_heatmap(df)


def remove_feature():
    df = st.session_state.df
    col_list = df.columns.tolist()
    st.subheader("Select the features to remove")
    features = st.multiselect("Features in dataframe are:", col_list)
    if st.button("Remove", key="remove_button"):
        df.drop(features, axis=1, inplace=True)
        st.session_state.df = df
        st.success(f"{features} removed successfully!")

def handle_missing_values():
    df = st.session_state.df
    st.subheader("Handling Missing Values")

    if 'missing_values_handled' not in st.session_state:
        st.session_state.missing_values_handled = False

    if st.session_state.missing_values_handled:
        st.success("Missing values have already been handled. This section is now disabled.")
    else:
        methods = st.selectbox("Select the method to handle missing values", ("Remove Rows with Missing Values", "Fill with Mean", "Fill with Median", "Fill with Mode"))
        if st.button("Apply", key="missing_values_button"):
            if methods == "Remove Rows with Missing Values":
                df = df.dropna()
                st.success("Rows with missing values removed successfully!")
            elif methods == "Fill with Mean":
                df = df.fillna(df.mean())
                st.success("Missing values filled with Mean successfully!")
            elif methods == "Fill with Median":
                df = df.fillna(df.median())
                st.success("Missing values filled with Median successfully!")
            elif methods == "Fill with Mode":
                df = df.fillna(df.mode().iloc[0])
                st.success("Missing values filled with Mode successfully!")

            st.session_state.df = df
            st.session_state.missing_values_handled = True
            # display()


def handle_outliers():
    df = st.session_state.df
    st.subheader("Handling Outliers")

    if 'outliers_removed' not in st.session_state:
        st.session_state.outliers_removed = False

    if st.session_state.outliers_removed:
        st.success("Outliers have already been removed. This section is now disabled.")
    else:
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
            st.session_state.outliers_removed = True

def categorical_feature_encoding():
    df = st.session_state.df
    st.subheader("Categorical Feature Encoding")
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    if len(categorical_columns) == 0:
        st.success("No categorical features found in the dataset.")
    else:
        encoding_methods = ["Label Encoding", "One Hot Encoding"]
        
        st.write("Select features for Label Encoding:")
        label_encoding_features = st.multiselect("Label Encoding", categorical_columns)

        st.write("Select features for One Hot Encoding:")
        one_hot_encoding_features = st.multiselect("One Hot Encoding", [col for col in categorical_columns if col not in label_encoding_features])

        if st.button("Apply Encoding", key="encoding_button"):
            if not label_encoding_features and not one_hot_encoding_features:
                st.error("No features selected for encoding.")
            else:
                try:
                    # Apply label encoding
                    for column in label_encoding_features:
                        df[column] = df[column].astype("category").cat.codes

                    # Apply one hot encoding
                    if one_hot_encoding_features:
                        df = pd.get_dummies(df, columns=one_hot_encoding_features, drop_first=True)

                    st.session_state.df = df
                    st.success("Categorical encoding applied successfully!")
                    display()
                except Exception as e:
                    st.error(f"Error in encoding: {e}")



def feature_engineering():
    df = st.session_state.df
    st.subheader("Feature Engineering")
    col_list = df.columns.tolist()
    selected_features = st.multiselect("Select the features to create new feature:", col_list)
    new_feature_name = st.text_input("Enter new feature name:")
    operation = st.selectbox("Select operation:", ["Sum", "Difference", "Product", "Division", "Custom"])
    # custom_operation = st.text_input("Enter custom operation (Python syntax):") if operation == "Custom" else ""

    if st.button("Create Feature", key="create_feature_button"):
        if len(selected_features) < 2 and operation != "Custom":
            st.error("Select at least two features to create a new feature.")
        else:
            try:
                if operation == "Sum":
                    df[new_feature_name] = df[selected_features].sum(axis=1)
                elif operation == "Difference":
                    df[new_feature_name] = df[selected_features[0]] - df[selected_features[1]]
                elif operation == "Product":
                    df[new_feature_name] = df[selected_features].prod(axis=1)
                elif operation == "Division":
                    df[new_feature_name] = df[selected_features[0]] / df[selected_features[1]]
                # elif operation == "Custom":
                #     df[new_feature_name] = df.apply(lambda row: eval(custom_operation, {"np": np, "pd": pd}, row), axis=1)
                # st.session_state.df = df
                st.success(f"New feature '{new_feature_name}' created successfully!")
                display()
            except Exception as e:
                st.error(f"Error in custom operation: {e}")

def split_data():
    df = st.session_state.df
    st.subheader("Splitting the dataset into features and target variable")

    if 'split_done' not in st.session_state:
        st.session_state.split_done = False

    if st.session_state.split_done:
        st.success("Dataset has already been split. This section is now disabled.")
    else:
        col1, col2 = st.columns(2)
        target = col1.selectbox("Select the target variable", df.columns.tolist())
        split_type = col2.selectbox("Select the type of split", ["Train and Test", "Train, Validation and Test"])

        test_size = st.slider("Select test size ratio", 0.1, 0.5, 0.3)
        validation_size = st.slider("Select validation size ratio (from remaining data after test split)", 0.1, 0.5, 0.2) if split_type == "Train, Validation and Test" else None

        if st.button("Apply", key="split"):
            st.session_state.split_done = True
            X = df.drop(target, axis=1)
            y = df[target]

            if split_type == "Train and Test":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.success("Dataset split successfully into Train and Test sets.")
            elif split_type == "Train, Validation and Test":
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.X_val = X_val
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_val = y_val
                st.success("Dataset split successfully into Train, Validation, and Test sets.")


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
            display_graphs()
            handle_missing_values()
            handle_outliers()
            remove_feature()
            categorical_feature_encoding()
            feature_engineering()
            split_data()
            model_training()
            model_evaluation()



if __name__ == '__main__':
    main()