import zipfile
import numpy as np
import os
import csv
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from processing import*
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

# 1. Unzipping the PDF Files
zip_path = 'C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\dataset.zip'
extract_path = 'C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Data\\'
preprocessed_path = 'C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Preprocessed\\'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    pdf_files = [name for name in zip_ref.namelist() if name.endswith('.pdf')]

# Main loop for processing each PDF file for extracting titles/abstracts

pdf_texts = {pdf_file: pdf_to_text(extract_path + pdf_file) for pdf_file in pdf_files}


documents_dict = {}

for index, (pdf_file, text) in enumerate(pdf_texts.items(), start=1):
    # Extract the title based on specific conditions or methods
    if pdf_file == "dataset/doc7.pdf":
        title = extract_title_specific_rounding(extract_path + pdf_file)
    elif pdf_file == "dataset/doc11.pdf":
        title = 'Integrated approach for high resolution surface characterization: coupling focused ion beam with micro and nano mechanical tests.'
    elif pdf_file == 'dataset/doc15.pdf':
        title = 'ANOMALOUS RADIATION INDUCED BY 1-300 KEV DEUTERON ION BEAM IMPLANTATION ON PALLADIUM AND TITANIUM'
    elif pdf_file == 'dataset/doc17.pdf':
        title = 'Screening Energy of the d+d Reaction in an Electron Plasma Deduced from Cooperative Colliding Reaction'
    else:
        title = extract_title_method1(extract_path + pdf_file)
        # If title is too short or looks off, try the general pdfplumber method
        if not title or len(title.split()) < 2 or title.lower() == 'abstract':
            title = extract_title_with_pdfplumber(extract_path + pdf_file)

    # Extract the abstract
    abstract = extract_abstract(text)

    # Add the extracted information to the dictionary
    documents_dict[index] = {
        'file_name': pdf_file,
        'title': title,
        'abstract': abstract
    }
    
    # Optionally print the results
    print(f"Document Index: {index}")
    print(f"PDF File: {pdf_file}")
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("=" * 50)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text, return_tokenized=True):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if return_tokenized:
        return tokens
    else:
        return ' '.join(tokens)

for pdf_file in pdf_files:
    full_path = os.path.join(extract_path, pdf_file)
    raw_text = pdf_to_text(full_path)
    preprocessed_text = preprocess_text(raw_text, False)
    
    # Save the preprocessed text to a .txt file
    preprocessed_file_name = pdf_file.replace('.pdf', '_preprocessed.txt')
    with open(os.path.join(preprocessed_path, preprocessed_file_name), 'w', encoding='utf-8') as f:
        f.write(preprocessed_text)
     

folder_path = preprocessed_path
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

texts = []
for file in files:
    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
        texts.append(f.read())

# Tokenize the texts
tokenized_texts = [text.split() for text in texts]

# 1. Create a Word2Vec instance
model_w2v = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)

# 2. Build the vocabulary
model_w2v.build_vocab(tokenized_texts)

# 3. Train the model
model_w2v.train(tokenized_texts, total_examples=model_w2v.corpus_count, epochs=10)
model_w2v.save("word2vec_model.model")

# Get a vector representation for each document by averaging word vectors
w2v_vectors = [np.mean([model_w2v.wv[word] for word in text], axis=0) for text in tokenized_texts]

# Prepare the data for Doc2Vec
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]

# Train a Doc2Vec model
model_d2v = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=100)
model_d2v.build_vocab(tagged_data)
model_d2v.train(tagged_data, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)
model_d2v.save("doc2vec_model.model")

# Get vectors for each document
d2v_vectors = [model_d2v.infer_vector(text.split()) for text in texts]

# Save Word2Vec vectors with titles
csv_path_w2v = 'C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Embeddings\\word2vec_vectors.csv'
with open(csv_path_w2v, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title'] + [f'Embedding_{i}' for i in range(len(w2v_vectors[0]))])
    for file, embedding in zip(files, w2v_vectors):
        writer.writerow([file] + embedding.tolist())

# Save Doc2Vec vectors with titles
csv_path_d2v = 'C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Embeddings\\doc2vec_vectors.csv'
with open(csv_path_d2v, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title'] + [f'Embedding_{i}' for i in range(len(d2v_vectors[0]))])
    for file, embedding in zip(files, d2v_vectors):
        writer.writerow([file] + embedding.tolist())

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Assuming the embeddings are loaded in the w2v_vectors and d2v_vectors lists
# If you've saved the embeddings to file and need to reload them, use:
# w2v_vectors = np.loadtxt('C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Embeddings\\word2vec_vectors.txt')
# d2v_vectors = np.loadtxt('C:\\Users\\kma\\Documents\\NYU\\2023-24 Senior Year\\Predictive Analytics\\HW2\\Embeddings\\doc2vec_vectors.txt')

# Randomly pick 5 documents
random_indices = random.sample(range(len(w2v_vectors)), 5)

# Initialize a dictionary to store the results
similar_docs = {}

for idx in random_indices:
    # Calculate cosine similarities for Word2Vec vectors
    similarities_w2v = cosine_similarity([w2v_vectors[idx]], w2v_vectors)[0]
    
    # Calculate cosine similarities for Doc2Vec vectors
    similarities_d2v = cosine_similarity([d2v_vectors[idx]], d2v_vectors)[0]

    # Get indices of top 3 similar documents (excluding the document itself) for Word2Vec
    top_indices_w2v = similarities_w2v.argsort()[-4:-1][::-1]
    
    # Get indices of top 3 similar documents (excluding the document itself) for Doc2Vec
    top_indices_d2v = similarities_d2v.argsort()[-4:-1][::-1]

    # Store the results
    similar_docs[idx] = {
        "Word2Vec": [(files[i], similarities_w2v[i]) for i in top_indices_w2v],
        "Doc2Vec": [(files[i], similarities_d2v[i]) for i in top_indices_d2v]
    }

# Display the results
for idx, data in similar_docs.items():
    print(f"For document {files[idx]}:")
    print("Top 3 similar documents (Word2Vec):")
    for doc, sim in data["Word2Vec"]:
        print(f"{doc} with similarity score: {sim:.4f}")
    print("Top 3 similar documents (Doc2Vec):")
    for doc, sim in data["Doc2Vec"]:
        print(f"{doc} with similarity score: {sim:.4f}")
    print("="*50)

from myPCA import reduce_dimensionality

# Convert list of vectors to a 2D array
d2v_array = np.vstack(d2v_vectors)

# Now, use the 2D array for dimensionality reduction
reduced_data, explained_variance = reduce_dimensionality(d2v_array, 2)

"""
print("Reduced Data Shape:", reduced_data.shape)
print(f"Total variance explained by 30 components: {explained_variance * 100:.2f}%")
"""
from processing import *

# Assuming 'reduced_data' is already defined and preprocessed


from processing import determine_k, get_representative_docs_titles

# Assuming 'reduced_data' is already prepared and available as per your dimensionality reduction

# Determine the optimal number of clusters
k = determine_k(reduced_data)

# Run KMeans clustering
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(reduced_data)

import numpy as np
import csv

# Assuming kmeans, reduced_data, and clusters are already defined as per your previous code
# Also assuming documents_dict is populated with your document data

# Calculate distances and find closest documents
closest_docs_per_cluster = {}
for cluster_num in range(k):
    centroid = kmeans.cluster_centers_[cluster_num]
    distances = np.linalg.norm(reduced_data[clusters == cluster_num] - centroid, axis=1)
    closest_indices = np.argsort(distances)[:3]  # Indices of 3 closest documents in this cluster

    # Use documents_dict to get title and abstract of closest documents
    closest_docs_info = []
    for idx in closest_indices:
        doc_index = np.where(clusters == cluster_num)[0][idx] + 1  # +1 because document indices start from 1
        doc_data = documents_dict[doc_index]
        title_abstract = doc_data['title'] + ': ' + doc_data['abstract']
        closest_docs_info.append(title_abstract)

    closest_docs_per_cluster[cluster_num] = closest_docs_info

# Write to CSV
with open('cluster_info.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cluster Number', 'Representative Documents'])
    for cluster_num, docs in closest_docs_per_cluster.items():
        writer.writerow([cluster_num, ' | '.join(docs)])

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=50)  # Increased marker size

# Optional: Add annotations
for i, point in enumerate(reduced_data):
    plt.text(point[0], point[1], str(i), fontsize=9)  # Add text with document index

# Add legend for clusters
plt.legend(*scatter.legend_elements(), title="Clusters")

# Optionally, add centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.legend()

# Add title and labels (optional)
plt.title('Cluster Visualization of Document Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Save the plot
plt.savefig('clusters.png', format='png')

# Show the plot
plt.show()

# Assuming 'documents_dict' contains your documents and 'clusters' contains cluster labels

cluster_texts = {}
for index, cluster_label in enumerate(clusters):
    # Get the document text (title + abstract)
    doc_text = documents_dict[index + 1]['title'] + ' ' + documents_dict[index + 1]['abstract']
    
    if cluster_label in cluster_texts:
        cluster_texts[cluster_label] += ' ' + doc_text
    else:
        cluster_texts[cluster_label] = doc_text

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(cluster_texts.values())

def extract_top_keywords(tfidf_matrix, feature_names, top_n=6):
    top_keywords = {}
    for cluster_num, row in enumerate(tfidf_matrix):
        indices = row.toarray()[0].argsort()[-top_n:][::-1]  # Sort and get top n indices
        features = [feature_names[i] for i in indices]
        top_keywords[cluster_num] = features
    return top_keywords

feature_names = vectorizer.get_feature_names_out()
top_keywords = extract_top_keywords(tfidf_matrix, feature_names)

existing_data = []
with open('cluster_info.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Assuming the first row is the header
    for row in reader:
        existing_data.append({headers[i]: value for i, value in enumerate(row)})

for data in existing_data:
    cluster_num = int(data['Cluster Number'])
    if cluster_num in top_keywords:
        data['Top Keywords'] = ', '.join(top_keywords[cluster_num])

with open('cluster_info.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers + ['Top Keywords'])  # Update headers if needed

    for data in existing_data:
        row = [data.get(header, '') for header in headers] + [data.get('Top Keywords', '')]
        writer.writerow(row)
"""
#ChatGPT topic modeling below:
cluster_data = {}

with open('cluster_info.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        cluster_num, rep_docs, top_keywords = row
        cluster_data[cluster_num] = {
            'rep_docs': rep_docs,
            'top_keywords': top_keywords.split(', ')
        }

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):
    # Lowercasing and removing special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords and lemmatization
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

for cluster, data in cluster_data.items():
    processed_text = process_text(data['rep_docs'])
    cluster_data[cluster]['processed_text'] = processed_text

# Count word frequencies per cluster
for cluster, data in cluster_data.items():
    data['word_freq'] = Counter(data['processed_text'])

# Identify unique and relevant keywords for each cluster
for cluster, data in cluster_data.items():
    # Compare word frequencies within the cluster against other clusters
    unique_keywords = set()
    for word, freq in data['word_freq'].items():
        is_unique = True
        for other_cluster, other_data in cluster_data.items():
            if other_cluster != cluster and word in other_data['word_freq']:
                is_unique = False
                break
        if is_unique:
            unique_keywords.add(word)

    # Update keywords by combining unique keywords with existing ones
    data['enhanced_keywords'] = list(unique_keywords) + data['top_keywords']

with open('enhanced_cluster_info.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cluster Number', 'Representative Documents', 'Top Keywords'])

    for cluster, data in cluster_data.items():
        writer.writerow([cluster, data['rep_docs'], ', '.join(data['enhanced_keywords'])])
"""