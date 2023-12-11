import os
import glob
import pandas as pd
import re

def combine_and_filter_entries(folder_path, output_folder, algorithm=None, filter_type=None, neighbourhood_size=None, similarity_threshold=None, include_negative_correlations=None):
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

   # Handle 'Start' and 'Elapsed time' columns
   combined_df['Elapsed time'] = combined_df.apply(lambda row: convert_elapsed_time(row), axis=1)
   combined_df.drop(columns=['Start'], inplace=True)

   # Filter the entries based on the specified parameters
   mask = True

   if algorithm:
      mask &= (combined_df['Algorithm'] == algorithm)
      
   if filter_type is not None:
      mask &= (combined_df['Filter Type'] == filter_type)

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
   filtered_file_name = f'filtered_results_{algorithm}_{neighbourhood_size}_{similarity_threshold}_{include_negative_correlations}_{filter_type}.csv'
   filtered_file_path = os.path.join(output_folder, filtered_file_name)
   filtered_df.to_csv(filtered_file_path, index=False)

   return filtered_df

def convert_elapsed_time(row):
   if pd.isnull(row['Start']):
      # Convert 'Elapsed time' to seconds
      minutes, seconds, milliseconds = map(int, re.findall(r'(\d+)', row['Elapsed time']))
      return f"{minutes*60 + seconds + milliseconds/1000:.3f}s"
   elif pd.isnull(row['Elapsed time']):
      # If 'Elapsed time' is NaN, convert 'Start' to seconds
      minutes, seconds, milliseconds = map(int, re.findall(r'(\d+)', row['Start']))
      return f"{minutes*60 + seconds + milliseconds/1000:.3f}s"
   else:
      # Keep 'Elapsed time' as is
      return row['Elapsed time']

def run_multiple_combinations(folder_path, output_folder):
   algorithms = ['Item-Based', 'User-Based']
   filter_types = ['Top-K Neighbours', 'Similarity Threshold', 'Combined Top-K & Similarity Threshold']
   neighbourhood_sizes = [None]
   similarity_thresholds = [None]
   include_negative_correlations = [True, False]

   for algorithm in algorithms:
      for filter_type in filter_types:
         for neighbourhood_size in neighbourhood_sizes:
               for similarity_threshold in similarity_thresholds:
                  for inc_neg_corr in include_negative_correlations:
                     print(f"Running combination: {algorithm}, {filter_type}, {neighbourhood_size}, {similarity_threshold}, {inc_neg_corr}")
                     combine_and_filter_entries(
                           folder_path,
                           output_folder,
                           algorithm=algorithm,
                           filter_type=filter_type,
                           neighbourhood_size=neighbourhood_size,
                           similarity_threshold=similarity_threshold,
                           include_negative_correlations=inc_neg_corr
                     )

# Example usage
folder_path = 'results'
output_folder = 'combined_results'
run_multiple_combinations(folder_path, output_folder)
