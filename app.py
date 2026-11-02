import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Data Cleaning and Visualization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Data Cleaning and Visualization Dashboard")
st.write(
    "An interactive dashboard for data preprocessing, "
    "outlier detection, visualization and data storytelling."
)

st.divider()

# 1. Upload Dataset
st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded dataset is empty.")
            st.stop()

        st.success("Dataset uploaded successfully!")

        # Reset stored results when a new file is uploaded
        if st.session_state.get("file_name") != uploaded_file.name:
            st.session_state["file_name"] = uploaded_file.name
            st.session_state.pop("cleaned_df", None)

        # 2. Dataset Overview
        st.header("2. Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Original Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # Dataset information
        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info, use_container_width=True)

        # 3. Missing Value Analysis
        st.header("3. Missing Value Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:

            fig, ax = plt.subplots(figsize=(10, 4))
            missing.plot(kind="bar", ax=ax)

            ax.set_title("Missing Values in Each Column")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Missing Values")
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            missing_buffer = BytesIO()
            fig.savefig(missing_buffer, format="png", bbox_inches="tight")
            missing_buffer.seek(0)

            st.download_button(
                "Download Missing Values Graph",
                missing_buffer,
                "missing_values.png",
                "image/png"
            )

            plt.close(fig)

        else:
            st.success("No missing values found!")

        # 4. Data Cleaning
        st.header("4. Data Cleaning")

        if st.button("Clean Dataset"):

            cleaned_df = df.copy()

            original_rows = len(cleaned_df)
            original_missing = int(cleaned_df.isnull().sum().sum())
            original_duplicates = int(cleaned_df.duplicated().sum())

            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates()

            # Handle missing values
            for column in cleaned_df.columns:

                if cleaned_df[column].isnull().sum() > 0:

                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):

                        median_value = cleaned_df[column].median()

                        if pd.notna(median_value):
                            cleaned_df[column] = cleaned_df[column].fillna(
                                median_value
                            )

                    else:

                        mode_value = cleaned_df[column].mode()

                        if not mode_value.empty:
                            cleaned_df[column] = cleaned_df[column].fillna(
                                mode_value.iloc[0]
                            )

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_stats"] = {
                "original_rows": original_rows,
                "original_missing": original_missing,
                "original_duplicates": original_duplicates
            }

            st.success("Dataset cleaned successfully!")

        # Show cleaning results
        if "cleaned_df" in st.session_state:

            cleaned_df = st.session_state["cleaned_df"].copy()
            stats = st.session_state["cleaning_stats"]

            st.subheader("Cleaned Dataset")
            st.dataframe(cleaned_df.head(10), use_container_width=True)

            # Before and After Comparison
            st.header("5. Before and After Cleaning")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Missing Values",
                int(cleaned_df.isnull().sum().sum()),
                delta=int(cleaned_df.isnull().sum().sum()) -
                      stats["original_missing"]
            )

            col2.metric(
                "Duplicate Rows",
                int(cleaned_df.duplicated().sum()),
                delta=int(cleaned_df.duplicated().sum()) -
                      stats["original_duplicates"]
            )

            col3.metric(
                "Rows Removed",
                stats["original_rows"] - len(cleaned_df)
            )

            # 6. Outlier Detection
            st.header("6. Outlier Detection")

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
                st.write("Lower Limit:", round(lower_limit, 2))
                st.write("Upper Limit:", round(upper_limit, 2))

                fig, ax = plt.subplots(figsize=(8, 4))

                sns.boxplot(
                    x=cleaned_df[selected_column],
                    ax=ax
                )

                ax.set_title("Outlier Detection: " + selected_column)
                st.pyplot(fig)

                box_buffer = BytesIO()
                fig.savefig(box_buffer, format="png", bbox_inches="tight")
                box_buffer.seek(0)

                st.download_button(
                    "Download Outlier Graph",
                    box_buffer,
                    "outlier_detection.png",
                    "image/png"
                )

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

            # 7. Data Visualization
            st.header("7. Data Visualization")

            if numeric_columns:

                st.subheader("Histogram")

                hist_column = st.selectbox(
                    "Choose a column for Histogram",
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

                hist_buffer = BytesIO()
                fig.savefig(hist_buffer, format="png", bbox_inches="tight")
                hist_buffer.seek(0)

                st.download_button(
                    "Download Histogram",
                    hist_buffer,
                    "histogram.png",
                    "image/png"
                )

                plt.close(fig)

                st.subheader("Bar Chart")

                bar_column = st.selectbox(
                    "Choose a column for Bar Chart",
                    cleaned_df.columns.tolist(),
                    key="bar"
                )

                fig, ax = plt.subplots(figsize=(8, 4))

                cleaned_df[bar_column].value_counts().head(10).plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title("Top 10 Values: " + bar_column)
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)

                bar_buffer = BytesIO()
                fig.savefig(bar_buffer, format="png", bbox_inches="tight")
                bar_buffer.seek(0)

                st.download_button(
                    "Download Bar Chart",
                    bar_buffer,
                    "bar_chart.png",
                    "image/png"
                )

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

                    heat_buffer = BytesIO()
                    fig.savefig(
                        heat_buffer,
                        format="png",
                        bbox_inches="tight"
                    )
                    heat_buffer.seek(0)

                    st.download_button(
                        "Download Correlation Heatmap",
                        heat_buffer,
                        "correlation_heatmap.png",
                        "image/png"
                    )

                    plt.close(fig)

                else:
                    st.info(
                        "At least two numerical columns are required "
                        "for a correlation heatmap."
                    )

            else:
                st.info("No numerical columns available for visualization.")

            # 8. Data Storytelling
            st.header("8. Key Findings and Data Storytelling")

            st.write(
                "The following findings are automatically generated "
                "from the uploaded dataset."
            )

            st.write(
                f"• The dataset contains {len(df)} original rows "
                f"and {len(df.columns)} columns."
            )

            st.write(
                f"• {stats['original_missing']} missing values "
                "were found before cleaning."
            )

            st.write(
                f"• {stats['original_duplicates']} duplicate rows "
                "were identified before cleaning."
            )

            st.write(
                f"• {len(cleaned_df)} rows remain after removing "
                "duplicate records."
            )

            st.write(
                f"• {int(cleaned_df.isnull().sum().sum())} "
                "missing values remain after cleaning."
            )

            # 9. Export Cleaned Dataset
            st.header("9. Export Cleaned Dataset")

            csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Cleaned Dataset",
                csv_data,
                "cleaned_dataset.csv",
                "text/csv"
            )

            # Export report
            report = f"""
DATA CLEANING AND VISUALIZATION REPORT

Original Rows: {stats['original_rows']}
Original Columns: {len(df.columns)}

Original Missing Values: {stats['original_missing']}
Original Duplicate Rows: {stats['original_duplicates']}

Rows After Cleaning: {len(cleaned_df)}
Remaining Missing Values: {int(cleaned_df.isnull().sum().sum())}
Remaining Duplicate Rows: {int(cleaned_df.duplicated().sum())}

This report summarizes the data cleaning and preprocessing results.
Visualizations include missing-value analysis, outlier detection,
histograms, bar charts and correlation heatmaps where applicable.
"""

            st.download_button(
                "Download Data Cleaning Report",
                report,
                "data_cleaning_report.txt",
                "text/plain"
            )

            st.success("Your cleaned dataset and report are ready!")

    except Exception as e:
        st.error("An error occurred while processing the dataset.")
        st.exception(e)

else:
    st.info("Please upload a CSV file to start the analysis.")

st.divider()

st.caption(
    "Data Cleaning and Visualization Project | "
    "Built using Python, Pandas, NumPy, Matplotlib, Seaborn and Streamlit"
)
