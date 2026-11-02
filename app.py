import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Data Cleaning and Visualization",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Data Cleaning and Visualization")
st.write(
    "Upload your dataset, clean the data, detect outliers "
    "and visualize the results."
)

st.divider()

# 1. Upload Dataset
st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # 2. Dataset Overview
    st.header("2. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("First Five Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(info, use_container_width=True)

    # 3. Missing Value Analysis
    st.header("3. Missing Value Analysis")

    missing = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values in Each Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # 4. Data Cleaning
    st.header("4. Data Cleaning")

    if st.button("Clean Dataset"):

        cleaned_df = df.copy()

        # Remove duplicate rows
        cleaned_df = cleaned_df.drop_duplicates()

        # Handle missing values
        for column in cleaned_df.columns:

            if cleaned_df[column].isnull().sum() > 0:

                if pd.api.types.is_numeric_dtype(cleaned_df[column]):

                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].median()
                    )

                else:

                    mode_value = cleaned_df[column].mode()

                    if not mode_value.empty:
                        cleaned_df[column] = cleaned_df[column].fillna(
                            mode_value[0]
                        )

        st.session_state["cleaned_df"] = cleaned_df

        st.success("Data cleaning completed successfully!")

    # Show cleaned data
    if "cleaned_df" in st.session_state:

        cleaned_df = st.session_state["cleaned_df"].copy()

        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_df, use_container_width=True)

        st.write(
            "Remaining Missing Values:",
            int(cleaned_df.isnull().sum().sum())
        )

        st.write(
            "Remaining Duplicate Rows:",
            int(cleaned_df.duplicated().sum())
        )

        # 5. Outlier Detection
        st.header("5. Outlier Detection")

        numeric_columns = cleaned_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select a numerical column",
                numeric_columns
            )

            Q1 = cleaned_df[selected_column].quantile(0.25)
            Q3 = cleaned_df[selected_column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            outliers = cleaned_df[
                (cleaned_df[selected_column] < lower_limit) |
                (cleaned_df[selected_column] > upper_limit)
            ]

            st.write("Number of Outliers:", len(outliers))

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.boxplot(
                x=cleaned_df[selected_column],
                ax=ax
            )

            ax.set_title("Outlier Detection")
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Apply IQR Outlier Capping"):

                cleaned_df[selected_column] = (
                    cleaned_df[selected_column].clip(
                        lower_limit, upper_limit
                    )
                )

                st.session_state["cleaned_df"] = cleaned_df

                st.success("Outliers capped successfully!")

        else:
            st.info("No numerical columns available.")

        # 6. Data Visualization
        st.header("6. Data Visualization")

        if numeric_columns:

            st.subheader("Histogram")

            hist_column = st.selectbox(
                "Select column for Histogram",
                numeric_columns,
                key="hist"
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.histplot(
                cleaned_df[hist_column].dropna(),
                kde=True,
                ax=ax
            )

            ax.set_title("Distribution of " + hist_column)

            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Bar Chart")

            bar_column = st.selectbox(
                "Select column for Bar Chart",
                numeric_columns,
                key="bar"
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            cleaned_df[bar_column].value_counts().head(10).plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Top Values of " + bar_column)

            st.pyplot(fig)
            plt.close(fig)

            # Correlation Heatmap
            st.subheader("Correlation Heatmap")

            if len(numeric_columns) >= 2:

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.heatmap(
                    cleaned_df[numeric_columns].corr(),
                    annot=True,
                    cmap="coolwarm",
                    ax=ax
                )

                ax.set_title("Correlation Between Variables")

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.info(
                    "At least two numerical columns are needed "
                    "for a correlation heatmap."
                )

        # 7. Download Cleaned Dataset
        st.header("7. Download Cleaned Dataset")

        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        st.success("Your cleaned dataset is ready!")

else:

    st.info("Please upload a CSV file to begin.")

st.divider()

st.caption(
    "Data Cleaning and Visualization Project | "
    "Built using Python, Pandas and Streamlit"
)
