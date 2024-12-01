#pip install numpy matplotlib scikit-learn pandas scipy

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# User input with desired language
language = input("Please enter the desired language (e.g., 'en' for English, 'fr' for French): ").lower()
your_path = "" # to change with your path

# Set the directory paths manually
input_directory = f'{your_path}/Style-Embeddings-paper-zip/DATA/'  # Replace with your actual input directory
output_directory = f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.3/'  # Replace with your actual output directory

# Define the input and output file paths dynamically based on the language input
input_file_name = f'stylo_df_grouped_{language}.xlsx'
output_file_name = f'groundtruth_results_{language}.xlsx'

# Define the full input and output paths
input_path = os.path.join(input_directory, input_file_name)
output_path = os.path.join(output_directory, output_file_name)

# Ensure the output directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the dataset
stylo_df_grouped = pd.read_excel(input_path)

# Exclude non-feature columns
non_feature_columns = ['author', 'id']
feature_columns = ["Structural", "Indexes", "Letters", "Punctuation", 
                   "TAG", "NER", "Function words", "Numbers"]

# Define the specific comparisons: (author1, author2)
comparisons = [('Queneau_ref', 'Queneau_gen'), ('Feneon_ref', 'Queneau_gen')]

# Define significance level thresholds
alpha_1 = 0.05
alpha_2 = 0.01

# Initialize a list to store t-test results
t_test_results = []

# Iterate over the specified author comparisons
for author1, author2 in comparisons:
    # Compare each feature (family) between the two authors
    for feature in feature_columns:
        # Filter data for each author
        data1 = stylo_df_grouped[stylo_df_grouped['author'] == author1][feature]
        data2 = stylo_df_grouped[stylo_df_grouped['author'] == author2][feature]
        delta = data1.mean() - data2.mean()

        # Perform t-test
        t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

        # Determine significance levels
        if p_value < alpha_2:
            significance = "Highly Significant"
        elif p_value < alpha_1:
            significance = "Significant"
        else:
            significance = "Not Significant"

        # Append the result to the list
        t_test_results.append({
            'Comparison': f'{author1} vs {author2}',
            'Feature': feature,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significance': significance,
            'Delta': delta
        })

# Convert the list of results into a DataFrame
t_test_results_df = pd.DataFrame(t_test_results)
t_test_results_df['p-value'] = t_test_results_df['p-value'].apply(lambda x: '{:.12f}'.format(x))

# Save the results to a CSV file
t_test_results_df.to_excel(output_path, index=False)
