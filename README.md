# Embedding Style Beyond Topics: Analyzing Dispersion Effects Across Language Models

This repository contains the code, data, and results accompanying our submission to COLING 2025.

## Setup
- All scripts in the `CODE` folder are executable. You must set the language to either `'FR'` or `'EN'` (or `'fr'`, `'en'`) at the beginning of each script.
- Ensure that the correct local paths are set before executing the code.
- The code allows for reproduction of results and can be adapted to test additional models and/or a different corpus using the provided input format.
  
## Repository Structure

### `DATA`
Contains all input data used for the analysis:
- Stylometry reports (based on [EnzoFleur's repository](https://github.com/EnzoFleur/style_embedding_evaluation)).
- Embeddings from French and English datasets.

### `RESULTS`
Contains the output data from the analysis:
- **Clustering Results**: Includes NMI, Purity, and Mean S scores, as detailed in Section 4.1 of the paper.
- **Dispersion Results**: Contains condition-wise analysis, centroid means, and statistical significance, as discussed in Section 4.2.
- **Embedding Style Interpretability**: Results for Section 4.3.

### `CODE`
Contains the main scripts to reproduce the results:
- **Text Generation**: Code for generating additional classes using GPT-4.
- **Clustering Analysis**: Code for Section 4.1.
- **Dispersion Analysis**: Full experimental setup for Section 4.2, covering all dimensionalities and 30 seed iterations.
- **Ground Truth Analysis**: Code for Section 4.3, including analysis of style and dispersion deltas and MSE additional experiment per class.

### `PLOT`
Contains all major plots for each section:
- **Section 4.1**: 2D PCA clustering projections, along with class frequency bar plots.
- **Section 4.2**: Contour plots for dispersion results (using sentence-transformers/all-MiniLM-L12-v2) in 2D UMAP, including sensitivity plots across different iterations for decision making.
- **Section 4.3**: Correlation plots comparing dispersion deltas with stylometric ground truth deltas, and analysis of dispersion results across four classes.

## Additional Information
For any additional results or clarifications, please contact us: evangelia.zve@lip6.fr


