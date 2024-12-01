#pip install numpy pandas matplotlib scikit-learn pickle5

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base paths for the datasets in French and English
your_path = ""

base_paths = {
    "FR": f"{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - FRENCH/",
    "EN": f"{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - ENGLISH/"
}

# Output paths for results and plots in French and English
results_paths = {
    "FR": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.1/',
    "EN": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.1/'
}

# Prompt user to select language
language = input("Choose language (FR/EN): ").upper()
if language not in base_paths:
    raise ValueError("Invalid language choice. Please select 'FR' or 'EN'.")

# Set base path and output directories based on the selected language
base_path = base_paths[language]
results_dir = results_paths[language]
output_file = os.path.join(results_dir, f"clustering_evaluation_results_{language}.xlsx")
plots_dir = os.path.join(results_dir, "plots")

# Ensure results and plots directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Function definitions
def purity_score(y_true, y_pred):
    """
    Calculate the purity score for the clustering results.
    """
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def calculate_nmi(y_true, y_pred):
    """
    Calculate the Normalized Mutual Information (NMI) score.
    """
    return normalized_mutual_info_score(y_true, y_pred)

def apply_kmeans(data, n_clusters=4):
    """
    Apply KMeans clustering to the given data.
    """
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(data)
    return labels + 1, model

def assign_group_labels(embeddings_dict):
    """
    Assign group labels based on the embeddings dictionary.
    """
    labels = []
    for group, embeddings in embeddings_dict.items():
        labels.extend([group] * len(embeddings))
    return np.array(labels)

def load_embeddings_from_pickle(embeddings_dir):
    """
    Load embeddings from a pickle file.
    """
    try:
        with open(embeddings_dir, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError as e:
        print(f"File not found: {embeddings_dir}")
        raise e

def cluster_and_evaluate(data, labels, method_name, dimensionality, model_name, results):
    """
    Perform clustering and evaluation, then save the results.
    """
    y_pred, _ = apply_kmeans(data)
    purity = purity_score(labels, y_pred)
    nmi = calculate_nmi(labels, y_pred)
    results.append((model_name, method_name, dimensionality, purity, nmi))

def global_score(purity, nmi):
    """
    Calculate a global score based on purity and NMI.
    """
    return (purity + nmi) / 2

# Simplified Model configurations
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

# Datasets paths
dataset_folders = {
    "Queneau_ref": "embeddings_Queneau_ref/",
    "Feneon_ref": "embeddings_Feneon_ref/",
    "Queneau_gen": "embeddings_Queneau_gen/",
    "Feneon_gen": "embeddings_Feneon_gen/"
}

# Initialize results storage
all_results = []

for config in model_configs:
    model_name = config['model_name']
    
    # Load the embeddings for each class
    embeddings_dict = {}
    for class_name, folder in dataset_folders.items():
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        embeddings_dir = os.path.join(base_path, folder, f"{safe_model_name}_embeddings.pkl")
        embeddings_dict[class_name] = load_embeddings_from_pickle(embeddings_dir)
    
    # Combine embeddings
    combined_embeddings = np.concatenate([embeddings_dict[key] for key in embeddings_dict.keys()])
    
    # Assign group labels
    group_labels = assign_group_labels(embeddings_dict)

    # Initialize results storage for current model
    results = []

    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(combined_embeddings)

    # FullD
    cluster_and_evaluate(embeddings_scaled, group_labels, 'FullD', 'FullD', model_name, results)

    # PCA transformations
    for dim in [2, 3, 5, 10]:
        pca = PCA(n_components=dim)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        cluster_and_evaluate(embeddings_pca, group_labels, f'{dim}D PCA', dim, model_name, results)
    
    # Append current model results to all results
    all_results.extend(results)

# Create a DataFrame to store all results
all_results_df = pd.DataFrame(all_results, columns=['Model', 'Method', 'Dimensionality', 'Purity', 'NMI'])

# Add the global score to the results DataFrame
all_results_df['Global Score'] = all_results_df.apply(lambda row: global_score(row['Purity'], row['NMI']), axis=1)

# Save the results to an Excel file
all_results_df.to_excel(output_file, index=False)

# Calculate and display the median global score per method
median_scores_per_model_method = all_results_df.groupby(['Method'])['Global Score'].median().reset_index()
median_scores_per_model_method.columns = ['Method', 'Median Global Score']

# Print the median results
print("Median Global Score per Method:")
print(median_scores_per_model_method.sort_values(by="Median Global Score"))

# Calculate and display the mean global score per method
mean_scores_per_model_method = all_results_df.groupby(['Method'])['Global Score'].mean().reset_index()
mean_scores_per_model_method.columns = ['Method', 'Mean Global Score']

# Print the mean results
print("\nMean Global Score per Method:")
print(mean_scores_per_model_method.sort_values(by="Mean Global Score"))
