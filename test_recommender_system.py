import os
import glob
import pandas as pd

def combine_and_filter_entries(folder_path, output_folder, algorithm=None, neighbourhood_size=None, similarity_threshold=None, include_negative_correlations=None):
    # Check if the combined file already exists
    combined_file_name = 'combined_results.csv'
    combined_file_path = os.path.join(output_folder, combined_file_name)

    if os.path.exists(combined_file_path):
        # If the combined file exists, read it directly
        combined_df = pd.read_csv(combined_file_path)
    else:
        # If the combined file doesn't exist, combine data from multiple files into a single DataFrame
        combined_data = []

        file_paths = glob.glob(os.path.join(folder_path, '*.txt'))

        for file_path in file_paths:
            # Skip files that contain "Logger file" in their names
            if 'Logger file' in file_path:
                continue

            with open(file_path, 'r') as file:
                lines = file.read().split('\n\n')

            data = []
            for entry in lines:
                if entry.strip():  # Skip empty entries
                    lines = entry.strip().split('\n')
                    entry_dict = dict([line.split(': ') for line in lines])
                    data.append(entry_dict)

            combined_data.extend(data)

        combined_df = pd.DataFrame(combined_data)

        # Convert relevant columns to numeric type
        numeric_columns = ['Neighbourhood Size', 'Similarity Threshold', 'Include Negative Correlations', 'Total predictions', 'Total under predictions (< 1)',
                            'Total over predictions (> 5)', 'Number of cases with no valid neighbours', 'Average neighbours used', 'Elapsed time',
                            'Time to do averages', 'Time to do similarities', 'Time to do predictions', 'Minimum filtered neighbourhood size',
                            'Maximum filtered neighbourhood size', 'Minimum neighbours size', 'Maximum neighbours size', 'Result']

        combined_df[numeric_columns] = combined_df[numeric_columns].apply(pd.to_numeric, errors='ignore')
        

        # Save the combined DataFrame to a new file
        combined_df.to_csv(combined_file_path, index=False)
    print(f"Original: {combined_df}")

    # Filter the entries based on the specified parameters
    mask = True

    if algorithm:
        mask &= (combined_df['Algorithm'] == algorithm)

    if neighbourhood_size is not None:
        mask &= (combined_df['Neighbourhood Size'] == neighbourhood_size)

    if similarity_threshold is not None:
        mask &= (combined_df['Similarity Threshold'] == similarity_threshold)

    if include_negative_correlations is not None:
        mask &= (combined_df['Include Negative Correlations'] == include_negative_correlations)

    # Ignore specific columns like 'File', 'Logger file', 'MAE'
    ignore_columns = ['File', 'Logger file', 'MAE']
    filtered_df = combined_df.loc[mask, combined_df.columns.difference(ignore_columns)]

    # Save the filtered DataFrame to a new file
    filtered_file_name = f'filtered_results_{algorithm}_{neighbourhood_size}_{similarity_threshold}_{include_negative_correlations}.csv'
    filtered_file_path = os.path.join(output_folder, filtered_file_name)
    filtered_df.to_csv(filtered_file_path, index=False)

    return filtered_df

# Example usage
folder_path = 'results'
output_folder = 'combined_results'
filtered_data = combine_and_filter_entries(folder_path, output_folder, algorithm='User-Based', neighbourhood_size=2, similarity_threshold=0.000000, include_negative_correlations=False)
print(f"Filtered: {filtered_data}")
