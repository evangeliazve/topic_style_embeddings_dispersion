#pip install numpy pandas pickle5 tqdm umap-learn scipy matplotlib seaborn scikit-learn

import os
import numpy as np
import pandas as pd
import pickle
import tqdm
import umap

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Paths for datasets and results
your_path = "/Users/evangeliazve/Documents"
base_paths = {
    "FR": f'{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - FRENCH/',
    "EN": f'{your_path}/Style-Embeddings-paper-zip/DATA/Four Classes Embeddings - ENGLISH/'
}
results_dirs = {
    "FR": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.3/',
    "EN": f'{your_path}/Style-Embeddings-paper-zip/RESULTS/Section 4.3/'
}

# Prompt to select language
language = input("Choose language (FR/EN): ").upper()
if language not in base_paths:
    raise ValueError("Invalid language choice. Please select 'FR' or 'EN'.")

base_path = base_paths[language]
results_dir = results_dirs[language]
os.makedirs(results_dir, exist_ok=True)

# Load stylometric datasets with features mapped to 8 categories
input_directory = f'{your_path}/Style-Embeddings-paper-zip/DATA/'
stylo_df_grouped = pd.read_excel(input_directory + "stylo_df_grouped_fr.xlsx")

# Dataset paths for each author
dataset_folders = {
    "Queneau_ref": "embeddings_Queneau_ref/",
    "Feneon_ref": "embeddings_Feneon_ref/",
    "Queneau_gen": "embeddings_Queneau_gen/",
    "Feneon_gen": "embeddings_Feneon_gen/"
}

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

# Assuming authors here is our 4 classes (queneau_ref, queneau_gen, feneon_ref, feneon_gen)
# Load embeddings using the author name from `stylo_df_grouped`
def load_embeddings_with_pickle(model_name, author_name, base_path):
    # Correct the model name to correct file paths
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Get the dataset folder for the author name (assuming the author name is the key in dataset_folders)
    dataset_folder = dataset_folders.get(author_name)
    if dataset_folder is None:
        print(f"No dataset folder found for author {author_name}")
        return None

    # Construct the file path for the embeddings
    file_path = os.path.join(base_path, dataset_folder, f"{safe_model_name}_embeddings.pkl")
    
    # Load the embeddings if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Embeddings loaded for {author_name} from {file_path}")
        return embeddings
    else:
        print(f"Embeddings file not found for {author_name} at {file_path}")
        return None

# Train SVR regressor and evaluate
def svr_regressor_train_test(x, y, test_size=0.3, kernel='rbf'):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    regressor = SVR(kernel=kernel)
    regressor.fit(X_train, Y_train)
    train_pred = regressor.predict(X_train)
    test_pred = regressor.predict(X_test)
    mse_train = mean_squared_error(Y_train, train_pred)
    mse_test = mean_squared_error(Y_test, test_pred)
    return mse_test, mse_train

# Evaluate style embeddings using SVR for each author
def style_embedding_evaluation(embeddings, grouped_features, kernel='rbf', test_size=0.3):
    author_results = {}
    
    # Iterate over each author in the grouped features
    for author in grouped_features['author'].unique():
        # Select the features for the current author
        author_features = grouped_features[grouped_features['author'] == author].drop(['id', 'author'], axis=1)
        
        # Get the corresponding embeddings
        author_embeddings = embeddings.get(author)
        if author_embeddings is None:
            print(f"Skipping {author} due to missing embeddings.")
            continue
        
        res_dict = {}
        # Iterate over feature families (grouped data)
        features_columns = ["Structural", "Indexes", "Letters", "Punctuation", 
                   "TAG", "NER", "Function words", "Numbers"]
        for feature_family in features_columns:
            y = np.array(author_features[feature_family])
            y = (y - np.mean(y)) / np.std(y)  # Standardize the target variable
            if np.isnan(y).any():
                continue
            
            mse_test, mse_train = svr_regressor_train_test(author_embeddings, y, test_size=test_size, kernel=kernel)
            res_dict[feature_family] = {"mse_test": mse_test, "mse_train": mse_train}
        
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        author_results[author] = res_df

    return author_results

# Iterate over model configurations and load embeddings for each author
all_model_results_all = []
for config in model_configs:
    model_name = config['model_name']
    
    embeddings_all = {}
    
    # Load embeddings for each author using their names
    for author in stylo_df_grouped['author'].unique():
        embeddings = load_embeddings_with_pickle(model_name, author, base_path)
        if embeddings is not None:
            embeddings_all[author] = embeddings
    
    # Store MSE results per family per seed
    aggregated_mse_results_all = {}

    # Apply UMAP and calculate MSE for each seed
    predefined_seeds = [
    42, 7, 19, 23, 1, 100, 56, 77, 89, 33,
    8, 13, 5, 21, 34, 99, 67, 18, 50, 81,
    45, 22, 74, 37, 58, 90, 16, 11, 29, 85
    ]

    for seed in predefined_seeds:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        umap_results_all = {}
        for author, embeddings in embeddings_all.items():
            if embeddings is not None:
                umap_results_all[author] = reducer.fit_transform(embeddings)
            else:
                print(f"Skipping UMAP for {author} due to missing embeddings.")
        
        res_df_all = style_embedding_evaluation(umap_results_all, stylo_df_grouped, kernel='rbf')

        # Aggregate results
        for author, mse_df in res_df_all.items():
            for family, row in mse_df.iterrows():
                key = family
                if key not in aggregated_mse_results_all:
                    aggregated_mse_results_all[key] = {'Queneau_ref': [], 'Queneau_gen': [], 'Feneon_ref': [], 'Feneon_gen': []}
                aggregated_mse_results_all[key][author].append(row['mse_test'])

    # Calculate mean MSE across all seeds for this model
    model_mse_results_all = []
    for family, mse_data in aggregated_mse_results_all.items():
        mean_mse_feneon_ref = np.mean(mse_data['Feneon_ref']) if mse_data['Feneon_ref'] else np.nan
        mean_mse_feneon_gen = np.mean(mse_data['Feneon_gen']) if mse_data['Feneon_gen'] else np.nan
        mean_mse_queneau_ref = np.mean(mse_data['Queneau_ref']) if mse_data['Queneau_ref'] else np.nan
        mean_mse_queneau_gen = np.mean(mse_data['Queneau_gen']) if mse_data['Queneau_gen'] else np.nan

        model_mse_results_all.append({
            'Model': model_name,
            'Family': family,
            'Queneau_ref MSE Mean': mean_mse_queneau_ref,
            'Queneau_gen MSE Mean': mean_mse_queneau_gen,
            'Feneon_ref MSE Mean': mean_mse_feneon_ref,
            'Feneon_gen MSE Mean': mean_mse_feneon_gen,
        })

    all_model_results_all.extend(model_mse_results_all)

# Convert to DataFrame for easier manipulation
all_model_df_all = pd.DataFrame(all_model_results_all)

# Save the MSE scores by author and model
output_file_path = os.path.join(results_dir, f"model_feature_mse_{language.lower()}.xlsx")
all_model_df_all.to_excel(output_file_path, index=False)

# Group by 'Family' and calculate the mean MSE for each class across all models
mean_mse_by_family = all_model_df_all.groupby('Family').agg({
    'Queneau_ref MSE Mean': 'mean',
    'Queneau_gen MSE Mean': 'mean',
    'Feneon_ref MSE Mean': 'mean',
    'Feneon_gen MSE Mean': 'mean',
}).reset_index()

# Calculate delta for the means
mean_mse_by_family['Delta MSE Mean (Queneau_ref/Queneau_gen)'] = (
    mean_mse_by_family['Queneau_ref MSE Mean'] - mean_mse_by_family['Queneau_gen MSE Mean']
)
mean_mse_by_family['Delta MSE Mean (Feneon_ref/Queneau_gen)'] = (
    mean_mse_by_family['Feneon_ref MSE Mean'] - mean_mse_by_family['Queneau_gen MSE Mean']
)

# Save the mean MSE results
output_mean_file_path = os.path.join(results_dir, f"mean_mse_by_feature_{language.lower()}.xlsx")
mean_mse_by_family.to_excel(output_mean_file_path, index=False)

###### Find MSE statistical difference between classes
## As our population here is very small (12 obervations), we are looking to see if normality and equal variances are satified. 

def perform_stat_tests(df):
    results = []

    for family, group in df.groupby('Family'):
        # Shapiro-Wilk test for normality
        try:
            if len(group['Queneau_ref MSE Mean']) >= 3:
                stat_queneau_ref, p_queneau_ref = stats.shapiro(group['Queneau_ref MSE Mean'])
            else:
                p_queneau_ref = 'Not enough data'

            if len(group['Feneon_ref MSE Mean']) >= 3:
                stat_feneon_ref, p_feneon_ref = stats.shapiro(group['Feneon_ref MSE Mean'])
            else:
                p_feneon_ref = 'Not enough data'

            if len(group['Queneau_gen MSE Mean']) >= 3:
                stat_queneau_gen, p_queneau_gen = stats.shapiro(group['Queneau_gen MSE Mean'])
            else:
                p_queneau_gen = 'Not enough data'

            # Levene's test for equal variances between Queneau_ref and Queneau_gen
            stat_levene, p_levene = stats.levene(group['Queneau_ref MSE Mean'], group['Queneau_gen MSE Mean'])

            # Append results
            results.append({
                'Family': family,
                'Shapiro-Wilk p-value (Queneau_ref)': p_queneau_ref,
                'Shapiro-Wilk p-value (Feneon_ref)': p_feneon_ref,
                'Shapiro-Wilk p-value (Queneau_gen)': p_queneau_gen,
                'Levene’s p-value (Queneau_ref vs Queneau_gen)': p_levene
            })

        except ValueError as e:
            print(f"Error for {family}: {e}")
    
    return pd.DataFrame(results)

# Perform the tests
check_results_df = perform_stat_tests(all_model_df_all)
# Save the final results to an Excel file
output_check_file_path = os.path.join(results_dir, f"mse_normality_variance_results_{language.lower()}.xlsx")
check_results_df.to_excel(output_check_file_path, index=False)

### Mann-Witney U test for non-parametric distributions

from scipy.stats import mannwhitneyu

mannwhitney_results = []

# Step 3: Perform Mann-Whitney U tests for each family and comparison
for family, group in all_model_df_all.groupby('Family'):
    
    # Mann-Whitney U Test: Queneau_ref vs Queneau_gen
    stat_queneau, p_value_queneau = mannwhitneyu(
        group['Queneau_ref MSE Mean'], 
        group['Queneau_gen MSE Mean'], 
        alternative='two-sided'
    )
    
    # Mann-Whitney U Test: Feneon_ref vs Queneau_gen
    stat_feneon_queneau, p_value_feneon_queneau = mannwhitneyu(
        group['Feneon_ref MSE Mean'], 
        group['Queneau_gen MSE Mean'], 
        alternative='two-sided'
    )
    
    # Store the results for both tests
    mannwhitney_results.append({
        'Family': family,
        'Queneau_ref vs Queneau_gen U_statistic': stat_queneau,
        'Queneau_ref vs Queneau_gen p_value': round(p_value_queneau, 6),
        'Feneon_ref vs Queneau_gen U_statistic': stat_feneon_queneau,
        'Feneon_ref vs Queneau_gen p_value': round(p_value_feneon_queneau, 6)
    })

# Convert the Mann-Whitney U test results to a DataFrame
mannwhitney_df = pd.DataFrame(mannwhitney_results)

# Merge mean MSE values with Mann-Whitney test results
final_results = pd.merge(mean_mse_by_family, mannwhitney_df, on='Family')

# Save the final results to an Excel file
output_combined_file_path = os.path.join(results_dir, f"mean_mse_mannwhitney_results_{language.lower()}.xlsx")
final_results.to_excel(output_combined_file_path, index=False)