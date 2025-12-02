# {
  # "image": "mcr.microsoft.com/devcontainers/universal:2",
  # "features": {}
# }

import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Spreadsheet Data Extractor", layout="wide")

st.title("📊 Spreadsheet Data Extractor")
st.write(
    "Upload a CSV or Excel file, filter and select the data you need, "
    "and download the result."
)


@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel into a pandas DataFrame."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    return df


def make_excel_download(df: pd.DataFrame) -> BytesIO:
    """Convert DataFrame to an in-memory Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="FilteredData")
    output.seek(0)
    return output


uploaded_file = st.file_uploader(
    "📁 Upload your spreadsheet",
    type=["csv", "xlsx", "xls"],
    help="Drag & drop or browse your computer for a CSV or Excel file.",
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.success("File loaded successfully!")

    # Show basic info
    with st.expander("🔍 Data preview & info", expanded=True):
        st.write("**Shape:**", df.shape)
        st.dataframe(df.head(20))

    st.sidebar.header("⚙️ Filters & Settings")

    # Choose columns to keep
    all_columns = list(df.columns)
    columns_to_keep = st.sidebar.multiselect(
        "Columns to include in the output",
        options=all_columns,
        default=all_columns,
    )

    if not columns_to_keep:
        st.warning("Select at least one column to include in the output.")
        st.stop()

    # Work on a copy with selected columns
    filtered_df = df[columns_to_keep].copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Column filters")

    # Build filters dynamically based on column types
    for col in columns_to_keep:
        col_data = filtered_df[col]

        with st.sidebar.expander(f"Filter: {col}", expanded=False):
            # Numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                if min_val == max_val:
                    st.write(f"Only one value: {min_val}")
                else:
                    selected_min, selected_max = st.slider(
                        "Range",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"{col}_num_slider",
                    )
                    mask = (filtered_df[col] >= selected_min) & (
                        filtered_df[col] <= selected_max
                    )
                    filtered_df = filtered_df[mask]

            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                min_date = col_data.min().date()
                max_date = col_data.max().date()
                start_date, end_date = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    key=f"{col}_date_range",
                )

                # streamlit returns a single date or tuple depending on usage
                if isinstance(start_date, tuple):
                    start_date, end_date = start_date

                mask = (filtered_df[col].dt.date >= start_date) & (
                    filtered_df[col].dt.date <= end_date
                )
                filtered_df = filtered_df[mask]

            # Everything else treated as categorical/text
            else:
                unique_vals = sorted(col_data.dropna().unique())
                if len(unique_vals) > 0:
                    selected_vals = st.multiselect(
                        "Keep only these values",
                        options=unique_vals,
                        default=unique_vals,
                        key=f"{col}_cat_multiselect",
                    )
                    if selected_vals:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                    else:
                        # If user deselects everything, no rows will remain
                        filtered_df = filtered_df.iloc[0:0]
                else:
                    st.write("No values to filter.")

    st.markdown("### ✅ Filtered result")

    st.write(f"**Rows:** {len(filtered_df)} | **Columns:** {len(filtered_df.columns)}")

    if len(filtered_df) == 0:
        st.warning("No data left after filtering. Adjust your filters.")
    else:
        st.dataframe(filtered_df)

        st.markdown("### 💾 Download your data")

        # CSV download
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="filtered_data.csv",
            mime="text/csv",
        )

        # Excel download
        excel_file = make_excel_download(filtered_df)
        st.download_button(
            label="Download as Excel",
            data=excel_file,
            file_name="filtered_data.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )
else:
    st.info("Upload a spreadsheet file to get started.")
