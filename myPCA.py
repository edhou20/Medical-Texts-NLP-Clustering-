import numpy as np
from sklearn.decomposition import PCA

def reduce_dimensionality(embeddings_matrix, n_components):  

    # Initialize PCA and set the number of components
    pca = PCA(n_components)

    # Fit and transform the data
    reduced_data = pca.fit_transform(embeddings_matrix)

    # Return the reduced data and the explained variance
    explained_variance = pca.explained_variance_ratio_.sum()

    return reduced_data, explained_variance
