import pandas as pd

def is_large_int_column(column):
    if column.dtypes == 'int64':
        max_length = column.astype(str).str.len().max()
        return max_length > 8
    return False

def generate_object_column_statistics(df):
    # Create an empty dictionary to store statistical data for object columns with numeric data
    object_column_statistics = {}

    # Categorical Column Analysis
    categorical_columns = df.select_dtypes(include=['object', 'category'])
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])

    for col_name in categorical_columns.columns:
        col_data = df[col_name]

        # Check if the column contains numeric data (excluding large integers)
        
            # Initialize the dictionary to store statistics for each unique value in the object column
        object_column_statistics[col_name] = {}

        unique_values = col_data.unique()
        for unique_value in unique_values:
            # Filter the DataFrame for the current unique value
            filtered_df = df[df[col_name] == unique_value]

            # Perform statistical analysis for the numeric columns for the current unique value
            statistics = filtered_df[numeric_columns.columns].astype(float).describe().round(2)
            median_value = filtered_df[numeric_columns.columns].astype(float).median().round(2)
            mode_value = filtered_df[numeric_columns.columns].astype(float).mode().iloc[0].round(2)
            std_value = filtered_df[numeric_columns.columns].astype(float).std().round(2)
            average_value = filtered_df[numeric_columns.columns].astype(float).mean().round(2).squeeze()

            # Store the statistical data for the current unique value in the object column
            object_column_statistics[col_name][unique_value] = {
                "Statistics": statistics,
                "Median": median_value,
                "Mode": mode_value,
                "Standard Deviation": std_value,
                "Average": average_value
            }

    return object_column_statistics

def generate_summary_story(df):
    # Identify column names and data types
    column_info = df.dtypes.reset_index()
    column_info.columns = ['Column Name', 'Data Type']

    # Basic analysis
    num_rows, num_cols = df.shape
    summary = f"Summary of the CSV Dataset:\nNumber of rows: {num_rows}\nNumber of columns: {num_cols}\n\n"

    # Column information
    summary += "Column Information:\n"
    summary += str(column_info) + "\n\n"

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        num_missing = missing_values.sum()
        summary += f"Warning: There are {num_missing} missing values in the dataset.\n\n"

    # Descriptive statistics for numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_columns.empty:
        summary += "Descriptive Statistics for Numeric Columns:\n"
        summary += "-----------------------------------\n"

        for col in numeric_columns:
            col_data = numeric_columns[col]
            summary += f"Column: {col}\n"
            summary += f"Minimum: {col_data.min():.2f}\n"
            summary += f"Maximum: {col_data.max():.2f}\n"
            summary += f"Mean: {col_data.mean():.2f}\n"
            summary += f"Median: {col_data.median():.2f}\n"
            summary += f"Mode: {col_data.mode().iloc[0]:.2f}\n"
            summary += f"Standard Deviation: {col_data.std():.2f}\n\n"

        # Correlation matrix for numeric columns
        correlation_matrix = numeric_columns.corr().round(2)
        summary += "Correlation Matrix for Numeric Columns:\n"
        summary += str(correlation_matrix) + "\n\n"

    # Generate statistics for object columns with numeric data
    object_column_statistics = generate_object_column_statistics(df)

    # Include object column statistics in the summary
    summary += "Statistics for Object Columns with Numeric Data:\n"
    summary += "----------------------------------------------\n"

    for col_name, unique_stats in object_column_statistics.items():
        summary += f"Column: {col_name}\n"
        for unique_value, stats in unique_stats.items():
            summary += f"\nUnique Value: {unique_value}\n"
            summary += f"Statistics:\n{stats['Statistics'].to_string()}\n"
            summary += "Averages:\n"
            for col, avg in stats['Average'].items():
                summary += f"{col}: {avg:.2f}\n"
            summary += "\n"

    # Date Column Analysis
    date_columns = df.select_dtypes(include=['datetime64'])
    for col_name in date_columns.columns:
        col_data = df[col_name]
        min_date = col_data.min()
        max_date = col_data.max()
        date_range = max_date - min_date
        summary += f"Date Analysis for '{col_name}' column:\n"
        summary += f"Minimum Date: {min_date}\n"
        summary += f"Maximum Date: {max_date}\n"
        summary += f"Time Range: {date_range}\n\n"

    return summary

def read_csv_and_generate_story(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Convert int64 columns with large integers to object type
    for col_name in df.select_dtypes(include='int64').columns:
        if is_large_int_column(df[col_name]):
            df[col_name] = df[col_name].astype(str)

    # Generate the factual story summary
    summary_story = generate_summary_story(df)
    return summary_story

def main():
    # Replace 'your_file_path.csv' with the path to your CSV file
    file_path = 'finalconsum.csv'
    summary_result = read_csv_and_generate_story(file_path)
    print(summary_result)
    output_file_path = 'summary_output.txt'

    # Write the summary to the output file
    with open(output_file_path, 'w') as file:
        file.write(summary_result)


if __name__ == "__main__":
    main()
