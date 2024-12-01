# pip install numpy pandas scipy umap-learn pickle5

import numpy as np
import pandas as pd
import os
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from scipy.stats import chi2
import umap

your_path = ""
# Paths for French and English datasets
base_paths = {
    "FR": f'{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - FRENCH/',
    "EN": f'{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - ENGLISH/'
}

results_dirs = {
    "FR": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.2/FRENCH/',
    "EN": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.2/ENGLISH/'
}

### => Prompt to select language ###
language = input("Choose language (FR/EN): ").upper()
if language not in base_paths:
    raise ValueError("Invalid language choice. Please select 'FR' or 'EN'.")

base_path = base_paths[language]
results_dir = results_dirs[language]
os.makedirs(results_dir, exist_ok=True)
# Datasets paths
dataset_folders = {
    "Queneau_ref": "embeddings_Queneau_ref/",
    "Feneon_ref": "embeddings_Feneon_ref/",
    "Queneau_gen": "embeddings_Queneau_gen/",
    "Feneon_gen": "embeddings_Feneon_gen/"
}

# Predefined UMAP dimensions and seeds
umap_dimensions = [2, 3, 5, 10]
predefined_seeds = [42, 7, 19, 23, 1, 100, 56, 77, 89, 33, 8, 
                    13, 5, 21, 34, 99, 67, 18, 50, 81, 45, 22, 74, 37, 58, 
                    90, 16, 11, 29, 85]

# Model configurations
model_configs = [
    {"model_name": "mistral-embed"},
    {"model_name": "text-embedding-3-small"},
    {"model_name": "voyage-2"},
    {"model_name": "paraphrase-multilingual-mpnet-base-v2"},
    {"model_name": "intfloat/e5-base-v2"},
    {"model_name": "all-roberta-large-v1"},
    {"model_name": "dangvantuan/sentence-camembert-base"},
    {"model_name": "OrdalieTech/Solon-embeddings-large-0.1"},
    {"model_name": "FacebookAI/xlm-roberta-large"},
    {"model_name": "distilbert/distilbert-base-uncased"},
    {"model_name": "sentence-transformers/all-MiniLM-L12-v2"},
    {"model_name": "intfloat/multilingual-e5-large"},
]

# Function to load embeddings from pickle files
def load_embeddings_from_pickle(embeddings_dir):
    try:
        with open(embeddings_dir, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError as e:
        print(f"File not found: {embeddings_dir}")
        raise e

# Function to calculate centroid and distances
def calculate_centroid_and_distances(embeddings):
    centroid = np.mean(embeddings, axis=0)
    distances = cdist(embeddings, [centroid]).flatten()
    return centroid, distances

# Function to calculate mean distances for all predefined seeds and UMAP dimensions
def calculate_mean_distances(embeddings_A1, embeddings_B1, embeddings_A2, embeddings_B2, n_components):
    all_distances_A1, all_distances_B1, all_distances_A2, all_distances_B2 = [], [], [], []
    
    for seed in predefined_seeds:
        reducer = umap.UMAP(n_components=n_components, random_state=seed)
        all_embeddings = np.concatenate((embeddings_A1, embeddings_B1, embeddings_A2, embeddings_B2), axis=0)
        reducer.fit(all_embeddings)
        
        transformed_A1 = reducer.transform(embeddings_A1)
        transformed_B1 = reducer.transform(embeddings_B1)
        transformed_A2 = reducer.transform(embeddings_A2)
        transformed_B2 = reducer.transform(embeddings_B2)
        
        _, distances_A1 = calculate_centroid_and_distances(transformed_A1)
        _, distances_B1 = calculate_centroid_and_distances(transformed_B1)
        _, distances_A2 = calculate_centroid_and_distances(transformed_A2)
        _, distances_B2 = calculate_centroid_and_distances(transformed_B2)
        
        all_distances_A1.append(distances_A1)
        all_distances_B1.append(distances_B1)
        all_distances_A2.append(distances_A2)
        all_distances_B2.append(distances_B2)
    
    mean_distances_A1 = np.mean([np.mean(d) for d in all_distances_A1])
    mean_distances_B1 = np.mean([np.mean(d) for d in all_distances_B1])
    mean_distances_A2 = np.mean([np.mean(d) for d in all_distances_A2])
    mean_distances_B2 = np.mean([np.mean(d) for d in all_distances_B2])

    return mean_distances_A1, mean_distances_B1, mean_distances_A2, mean_distances_B2, np.array(all_distances_A1), np.array(all_distances_B1), np.array(all_distances_A2), np.array(all_distances_B2)

# Main execution
all_model_distances = {class_name: [] for class_name in dataset_folders.keys()}

for config in model_configs:
    model_name = config['model_name']
    
    # Load the embeddings for each class
    embeddings_dict, distances_dict = {}, {}
    for class_name, folder in dataset_folders.items():
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        embeddings_dir = os.path.join(base_path, folder, f"{safe_model_name}_embeddings.pkl")
        embeddings_dict[class_name] = load_embeddings_from_pickle(embeddings_dir)
        
        # Calculate distances from the centroid
        _, distances = calculate_centroid_and_distances(embeddings_dict[class_name])
        distances_dict[class_name] = distances
    
    embeddings_A1, embeddings_B1 = embeddings_dict["Queneau_ref"], embeddings_dict["Feneon_ref"]
    embeddings_A2, embeddings_B2 = embeddings_dict["Queneau_gen"], embeddings_dict["Feneon_gen"]
    
    results = []
    for n_components in umap_dimensions:
        mean_dist_A1, mean_dist_B1, mean_dist_A2, mean_dist_B2, all_distances_A1, all_distances_B1, all_distances_A2, all_distances_B2 = calculate_mean_distances(
            embeddings_A1, embeddings_B1, embeddings_A2, embeddings_B2, n_components
        )

        if n_components == 2:
            for class_name, distances in zip(dataset_folders.keys(), [all_distances_A1, all_distances_B1, all_distances_A2, all_distances_B2]):
                all_model_distances[class_name].append(np.mean(distances, axis=0))
        
        conditions = [
            ('Feneon_gen', 'Queneau_ref', all_distances_B2, all_distances_A1, 'TOPIC'),
            ('Feneon_ref', 'Queneau_gen', all_distances_B1, all_distances_A2, 'TOPIC'),
            ('Queneau_ref', 'Queneau_gen', all_distances_A1, all_distances_A2, 'STYLE'),
            ('Feneon_gen', 'Feneon_ref', all_distances_B2, all_distances_B1, 'STYLE'),
            ('Feneon_ref', 'Queneau_ref', all_distances_B1, all_distances_A1, 'GLOBAL')
        ]
        
        for group1_name, group2_name, distances1, distances2, cond_type in conditions:
            t_stat, p_value = ttest_ind(np.hstack(distances1), np.hstack(distances2), equal_var=False)
            mean1 = np.mean(np.hstack(distances1))
            mean2 = np.mean(np.hstack(distances2))
            condition_satisfied = mean1 > mean2
            results.append({
                'Model': model_name,
                'UMAP_Dimension': f'UMAP_{n_components}D',
                'Condition': f'{group1_name} > {group2_name}',
                'Condition_Type': cond_type,
                'Group1': group1_name,
                'Group2': group2_name,
                'Mean1': mean1,
                'Mean2': mean2,
                'Condition_Satisfied': condition_satisfied,
                'T_Statistic': t_stat,
                'P_Value_T': p_value,
                'Condition_Significant': p_value < 0.05 if not np.isnan(p_value) else False
            })

    results_df = pd.DataFrame(results)
    # Save results per model
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    results_file = os.path.join(results_dir, f'results_{safe_model_name}_global_{language}.xlsx')
    results_df.to_excel(results_file, index=False)
    print(f"Results for model {model_name} saved to {results_file}")

# Calculate and save the mean distances across all models for UMAP 2D only
mean_distances_data = []
for class_name, distances_list in all_model_distances.items():
    distances_array = np.array(distances_list)
    mean_distances = np.mean(distances_array, axis=0)
    for idx, mean_distance in enumerate(mean_distances):
        mean_distances_data.append({
            'Class': class_name,
            'Text_Index': idx,
            'Mean_Distance_From_Centroid': mean_distance
        })

mean_distances_df = pd.DataFrame(mean_distances_data)
mean_distances_file = os.path.join(results_dir, f'distance_pertext_umap_2d_{language}.xlsx')
mean_distances_df.to_excel(mean_distances_file, index=False)

# Combine all results into one final file
all_results = []
for file in os.listdir(results_dir):
    if file.endswith(f'global_{language}.xlsx') and 'results' in file:
        file_path = os.path.join(results_dir, file)
        df = pd.read_excel(file_path)
        all_results.append(df)

final_results_df = pd.concat(all_results, ignore_index=True)
final_results_file = os.path.join(results_dir, f'dispersion_results_{language}.xlsx')
final_results_df.to_excel(final_results_file, index=False)

## Mean & Median Model evaluation 
final_results_file = os.path.join(results_dir, f'dispersion_results_{language}.xlsx')
mean_distances_file = os.path.join(results_dir, f'distance_pertext_umap_2d_{language}.xlsx')

# Load the combined results and mean distances
final_results_df = pd.read_excel(final_results_file)
mean_distances_df = pd.read_excel(mean_distances_file)

# Calculate mean of mean distances for each condition/dimension
mean_distances_summary = final_results_df.groupby(['UMAP_Dimension', 'Condition']) \
    .agg({
        'Mean1': 'mean',
        'Mean2': 'mean',
        'Condition_Satisfied': 'mean',
        'Condition_Significant': 'mean',
        'P_Value_T': lambda x: chi2.sf(-2 * np.sum(np.log(x)), 2 * len(x))
    }).reset_index()

# Calculate median of mean distances for each condition/dimension
median_mean_distances = final_results_df.groupby(['UMAP_Dimension', 'Condition']) \
    .agg({
        'Mean1': 'median',
        'Mean2': 'median',
        'Condition_Satisfied': 'median',
        'Condition_Significant': 'median',
        'P_Value_T': lambda x: chi2.sf(-2 * np.sum(np.log(x)), 2 * len(x))
    }).reset_index()

# Determine if medians and averages are satisfied and significant
def determine_satisfaction_and_significance(df):
    df['Condition_Satisfied'] = df['Mean1'] > df['Mean2']
    df['Condition_Significant'] = df['P_Value_T'] < 0.05
    
    return df


mean_distances_summary = determine_satisfaction_and_significance(mean_distances_summary)
median_mean_distances = determine_satisfaction_and_significance(median_mean_distances)

# Save the results to new files
mean_distances_summary_file = os.path.join(results_dir, f'mean_distances_summary_{language}.xlsx')
median_mean_distances_file = os.path.join(results_dir, f'median_mean_distances_{language}.xlsx')

mean_distances_summary.to_excel(mean_distances_summary_file, index=False)
median_mean_distances.to_excel(median_mean_distances_file, index=False)

print(f"Mean distances summary saved to {mean_distances_summary_file}")
print(f"Median mean distances saved to {median_mean_distances_file}")

#### Optimal Dimensionality 

# Group the final results by UMAP Dimension and calculate counts
grouped_df = final_results_df.groupby('UMAP_Dimension').agg(
    Conditions_Satisfied=('Condition_Satisfied', 'sum'),
    Significant_Conditions=('Condition_Significant', 'sum'),
    Total_Lines=('Condition_Satisfied', 'size'),
    Not_Satisfied_And_Significant=('Condition_Significant', lambda x: ((final_results_df['Condition_Satisfied'] == False) & x).sum())
).reset_index()

# Calculate the ratios for each dimensionality
grouped_df['Ratio_Conditions_Satisfied'] = grouped_df['Conditions_Satisfied'] / grouped_df['Total_Lines']

# Calculate the ratio of significant conditions to satisfied conditions
grouped_df['Ratio_Significant_to_Satisfied'] = grouped_df.apply(
    lambda row: row['Significant_Conditions'] / row['Conditions_Satisfied'] 
    if row['Conditions_Satisfied'] > 0 else 0, axis=1
)

# Calculate the number of not satisfied conditions
grouped_df['Not_Satisfied_Conditions'] = grouped_df['Total_Lines'] - grouped_df['Conditions_Satisfied']

# Calculate the ratio of significant conditions that are not satisfied
grouped_df['Ratio_Significant_to_Not_Satisfied'] = grouped_df.apply(
    lambda row: row['Not_Satisfied_And_Significant'] / row['Not_Satisfied_Conditions'] 
    if row['Not_Satisfied_Conditions'] > 0 else 0, axis=1
)

# Save the results to an Excel file
ratios_file = os.path.join(results_dir, f'ratios_per_dimensionality_{language}.xlsx')
grouped_df.to_excel(ratios_file, index=False)

