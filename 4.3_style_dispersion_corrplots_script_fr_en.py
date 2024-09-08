
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#### INPUT PATHS (FR & EN)

# Load the Stylometry report for 8 categories
# FR PATH
stylo_df_fr_file_path = '/your_path/Style-Embeddings-paper-zip/DATA/stylo_df_grouped_fr.xlsx'
# EN PATH
stylo_df_en_file_path = '/your_path/Style-Embeddings-paper-zip/DATA/stylo_df_grouped_en.xlsx'

stylo_df_grouped = pd.read_excel("CHOOSE LANGUAGE PATH FROM ABOVE")

# Load the mean distances from the previously saved file from section 4.2
# FR PATH
mean_distances_fr_file_path = '/your_path/Style-Embeddings-paper-zip/RESULTS/Section 4.2/mean_distances_across_models_umap_2d_FR.xlsx'
# EN PATH
mean_distances_en_file_path = '/your_path/Style-Embeddings-paper-zip/RESULTS/Section 4.2/mean_distances_across_models_umap_2d_EN.xlsx'

mean_distances_df = pd.read_excel("CHOOSE LANGUAGE PATH FROM ABOVE")


#### DISPERSION ANALYSIS BY CLASS (4 CLASSES)

# Create the 'text_id' by using the modulo operation
range_limit = 73
stylo_df_grouped['text_id'] = [i % range_limit for i in range(len(stylo_df_grouped))]

# Merging style features with dispersion metrics
df_merged = pd.merge(mean_distances_df, stylo_df_grouped, left_on=['Class', 'Text_Index'], right_on=['author', 'text_id'])

# Get unique authors (classes)
authors = df_merged['Class'].unique()

# Lists to store correlation and p-value results per author
correlation_data = []
p_value_data = []

# Function to calculate correlations and p-values
def calculate_correlation_and_pvalues(dataframe):
    corr_matrix = dataframe.corr()
    p_matrix = dataframe.corr().copy()
    
    # Calculate p-values for each pair of columns
    for col1 in dataframe.columns:
        for col2 in dataframe.columns:
            if col1 == col2:
                p_matrix.loc[col1, col2] = np.nan  # Diagonal is NaN
            else:
                _, p_value = pearsonr(dataframe[col1], dataframe[col2])
                p_matrix.loc[col1, col2] = p_value
    return corr_matrix, p_matrix

# Iterating over each author to calculate correlation
for author in authors:
    # Filter the data for the current author
    author_data = df_merged[df_merged['Class'] == author]
    
    # Ensure only numeric columns are used for correlation, exclude 'id' and 'Other Features'
    numeric_columns = author_data.select_dtypes(include=['float64', 'int64']).drop(columns=['text_id', "Text_Index",'Other Features'], errors='ignore')
    
    # Calculate the correlation matrix and p-values for the current author's data
    correlation_matrix, p_matrix = calculate_correlation_and_pvalues(numeric_columns)
    
    # Exclude 'Mean_Distance_From_Centroid' from the x-axis
    correlation_with_centroid = correlation_matrix['Mean_Distance_From_Centroid'].drop('Mean_Distance_From_Centroid')
    p_values_with_centroid = p_matrix['Mean_Distance_From_Centroid'].drop('Mean_Distance_From_Centroid')
    
    # Append the correlation and p-values data with the author's name
    correlation_data.append(pd.Series(correlation_with_centroid, name=author))
    p_value_data.append(pd.Series(p_values_with_centroid, name=author))

# Create a DataFrame where each row is an author and columns are correlations with stylistic features
correlation_df = pd.DataFrame(correlation_data)
p_values_df = pd.DataFrame(p_value_data)

# 1. Heatmap: Order by "Queneau_ref" correlations, include "Queneau_ref", "Queneau_gen", "Feneon_ref", "Feneon_gen"
authors_group = ["Feneon_gen", "Feneon_ref", "Queneau_ref", "Queneau_gen"]

# Sort features by "Queneau_ref" from highest to lowest correlation
correlation_sorted = correlation_df.loc["Feneon_gen"].sort_values(ascending=False).index
correlation_df_sorted = correlation_df[correlation_sorted]
p_values_df_sorted = p_values_df[correlation_sorted]

# Filter for the required authors
corr_group = correlation_df_sorted.loc[authors_group]
pval_group = p_values_df_sorted.loc[authors_group]

# Create the heatmap
fig, ax = plt.subplots(figsize=(10, 4))
heatmap = sns.heatmap(corr_group, 
                      annot=False, 
                      cmap='coolwarm', 
                      vmin=-1, 
                      vmax=1, 
                      center=0, 
                      linewidths=2, 
                      linecolor='white', 
                      cbar_kws={"shrink": 1},  # Control color bar size
                      ax=ax)

# Function to annotate the heatmap with p-values in asterisks format
def annotate_heatmap_with_pvalues(corr_matrix, p_matrix, ax, annot_kws=None):
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            corr_value = corr_matrix[i, j]
            p_value = p_matrix[i, j]
            if p_value < 0.01:
                text = f"{corr_value:.2f}**"
            elif p_value < 0.05:
                text = f"{corr_value:.2f}*"
            else:
                text = f"{corr_value:.2f}"
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=12)

# Annotate the heatmap with the correlation values and asterisks based on p-values
annotate_heatmap_with_pvalues(corr_group.values, pval_group.values, ax=ax)

# Set the rotation for the x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)

# Set the rotation for the y-axis labels
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=14)

# Adjust colorbar tick font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)  # Increase the colorbar font size here

# Show the plot
plt.tight_layout()
plt.show()

#### DISPERSION DELTA ANALYSIS FOR 2 COMPARISONS
## Queneau_ref vs Queneau_gen

# Dispersion delta
queneau_ref_distances = mean_distances_df[mean_distances_df['Class'] == 'Queneau_ref']['Mean_Distance_From_Centroid'].values
queneau_gen_distances = mean_distances_df[mean_distances_df['Class'] == 'Queneau_gen']['Mean_Distance_From_Centroid'].values

# List to store the results
difference_results = []

# Calculate the difference between each Queneau_ref distance and each Queneau_gen distance
for i, ref_dist in enumerate(queneau_ref_distances):
    for j, gen_dist in enumerate(queneau_gen_distances):
        difference = ref_dist - gen_dist
        difference_results.append({
            'Queneau_ref_Index': i,
            'Queneau_gen_Index': j,
            'Difference': difference
        })

# Convert the list to a DataFrame
difference_df_cond1 = pd.DataFrame(difference_results)

# Style delta calculation
queneau_ref_df = stylo_df_grouped[stylo_df_grouped['author'] == 'Queneau_ref']
queneau_gen_df = stylo_df_grouped[stylo_df_grouped['author'] == 'Queneau_gen']

# List of stylistic features
features = ['Function words', 'Letters', 'Numbers', 'TAG', 'NER', 'Structural', 'Punctuation', 'Indexes']

# Initialize a list to store the results
difference_results_stylo = []

for feature in features:
    for i, ref_val in enumerate(queneau_ref_df[feature].values):
        for j, gen_val in enumerate(queneau_gen_df[feature].values):
            difference = ref_val - gen_val
            difference_results_stylo.append({
                'Feature': feature,
                'Queneau_ref_Index': i,
                'Queneau_gen_Index': j,
                'Difference': difference
            })

difference_df_stylo_cond1 = pd.DataFrame(difference_results_stylo)


## Feneon_ref vs Queneau_gen

# Dispersion delta 
feneon_ref_distances = mean_distances_df[mean_distances_df['Class'] == 'Feneon_ref']['Mean_Distance_From_Centroid'].values
queneau_gen_distances = mean_distances_df[mean_distances_df['Class'] == 'Queneau_gen']['Mean_Distance_From_Centroid'].values

# List to store the results
difference_results = []

# Calculate the difference between each Queneau_ref distance and each Queneau_gen distance
for i, ref_dist in enumerate(feneon_ref_distances):
    for j, gen_dist in enumerate(queneau_gen_distances):
        difference = ref_dist - gen_dist
        difference_results.append({
            'Feneon_ref_Index': i,
            'Queneau_gen_Index': j,
            'Difference': difference
        })

# Convert the list to a DataFrame
difference_df_cond2 = pd.DataFrame(difference_results)


# Style Delta
feneon_ref_df = stylo_df_grouped[stylo_df_grouped['author'] == 'Feneon_ref']
queneau_gen_df = stylo_df_grouped[stylo_df_grouped['author'] == 'Queneau_gen']

# List of stylistic features
features = ['Function words', 'Letters', 'Numbers', 'TAG', 'NER', 'Structural', 'Punctuation', 'Indexes']

# Initialize a list to store the results
difference_results_stylo = []

for feature in features:
    for i, ref_val in enumerate(feneon_ref_df[feature].values):
        for j, gen_val in enumerate(queneau_gen_df[feature].values):
            difference = ref_val - gen_val
            difference_results_stylo.append({
                'Feature': feature,
                'Feneon_ref_Index': i,
                'Queneau_gen_Index': j,
                'Difference': difference
            })

difference_df_stylo_cond2 = pd.DataFrame(difference_results_stylo)


# Function to calculate Pearson correlation and p-values with significance asterisks
def calculate_correlations_with_significance(difference_df_stylo, difference_df):
    correlation_results = {}
    p_value_results = {}

    for feature in difference_df_stylo["Feature"].unique():
        filtered_stylo_df = difference_df_stylo[difference_df_stylo["Feature"] == feature]
        
        # Calculate Pearson correlation and p-value
        correlation, p_value = pearsonr(filtered_stylo_df["Difference"], difference_df["Difference"])
        correlation_results[feature] = correlation
        
        # Assign asterisk based on p-value significance
        if p_value < 0.01:
            p_value_results[feature] = '**'
        elif p_value < 0.05:
            p_value_results[feature] = '*'
        else:
            p_value_results[feature] = ''
    
    correlation_df = pd.DataFrame({
        'Feature': list(correlation_results.keys()),
        'Correlation': list(correlation_results.values()),
        'P-value': list(p_value_results.values())  # Store significance asterisks
    })

    return correlation_df

# Recalculate correlations with the updated function
correlation_df_cond1 = calculate_correlations_with_significance(difference_df_stylo_cond1, difference_df_cond1)
correlation_df_cond2 = calculate_correlations_with_significance(difference_df_stylo_cond2, difference_df_cond2)

# Sort by the highest to lowest correlation in Condition 1 (Queneau_ref/Queneau_gen)
correlation_df_cond1 = correlation_df_cond1.sort_values(by='Correlation', ascending=False)

# Merge the two dataframes for heatmap plotting, ensuring sorted features
merged_df = pd.merge(correlation_df_cond1[['Feature', 'Correlation', 'P-value']],
                     correlation_df_cond2[['Feature', 'Correlation', 'P-value']],
                     on='Feature', suffixes=('_Cond1', '_Cond2'))

# Create the heatmap data: Sort based on 'Correlation_Cond1' (Queneau_ref/Queneau_gen)
merged_df = merged_df.sort_values(by='Correlation_Cond1', ascending=False)

# Rearrange the order of the features based on the sorted correlation in Queneau_ref/Queneau_gen
sorted_features = merged_df['Feature'].tolist()

# Create heatmap data with sorted columns and invert the rows (swap positions)
heatmap_data = merged_df[['Correlation_Cond2', 'Correlation_Cond1']].T  # Swap the columns for conditions
heatmap_data.columns = sorted_features  # Ensure sorted order in heatmap

# Annotate the heatmap with asterisks for significance
annotations = np.array([merged_df['Correlation_Cond2'].astype(str) + merged_df['P-value_Cond2'], 
                        merged_df['Correlation_Cond1'].astype(str) + merged_df['P-value_Cond1']])

# Plotting the heatmap with swapped axes (Conditions on y-axis, Features on x-axis)
fig, ax = plt.subplots(figsize=(12, 3))
heatmap = sns.heatmap(heatmap_data, 
                      annot=False, 
                      cmap='coolwarm', 
                      vmin=-1, 
                      vmax=1, 
                      center=0, 
                      linewidths=2, 
                      linecolor='white', 
                      cbar_kws={"shrink": 1},  # Adjust color bar size to fit the plot
                      ax=ax)

# Function to annotate the heatmap with p-values and correlation values in asterisks format
def annotate_heatmap_with_pvalues(corr_matrix, p_matrix, ax, annot_kws=None):
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            corr_value = corr_matrix[i, j]
            p_value = p_matrix[i, j]
            if p_value.endswith('**'):
                text = f"{float(corr_value):.2f}**"
            elif p_value.endswith('*'):
                text = f"{float(corr_value):.2f}*"
            else:
                text = f"{float(corr_value):.2f}"
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=14)

# Annotate the heatmap with the correlation values and asterisks based on p-values
annotate_heatmap_with_pvalues(heatmap_data.values, annotations, ax=ax)

# Set the custom y-axis labels with the correct order after swapping
ax.set_yticklabels([r'$\Delta d(\text{Feneon\_ref}, \text{Queneau\_gen})$', 
                    r'$\Delta d(\text{Queneau\_ref}, \text{Queneau\_gen})$'], 
                   rotation=0, ha="right", fontsize=16)

# Set the rotation for the x-axis labels (features)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
ax.set_xlabel('')

# Adjust colorbar tick font size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)

# Show the plot
plt.tight_layout()
plt.show()



