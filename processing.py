import PyPDF2
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance

def determine_k(data):
    distortions = []
    max_k = min(len(data), 15)
    K = range(1, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    k_optimal = 1
    for i in range(1, len(distortions) - 1):
        if distortions[i] < distortions[i-1]:
            k_optimal = i + 1
        else:
            break

    return k_optimal

def get_representative_docs_titles(data, titles, clusters, centroids, n=3):
    representative_titles = []
    for i, centroid in enumerate(centroids):
        dist = [distance.euclidean(point, centroid) for j, point in enumerate(data) if clusters[j] == i]
        closest_indices = np.argsort(dist)[:n]
        cluster_titles = [titles[j] for j in closest_indices]
        representative_titles.append(cluster_titles)
    return representative_titles



# 2. Converting PDF to Text
def pdf_to_text(pdf_file_path):
    try:
        with open(pdf_file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
        return text
    except Exception as e:
        print(f"Error processing {pdf_file_path}: {e}")
        return ""

# 3. Extracting Title

def extract_title_specific_rounding(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        words = first_page.extract_words()

        # Filter out words that appear after the first 300 characters
        text_so_far = ''
        filtered_words = []
        for word in words:
            text_so_far += word['text']
            if len(text_so_far) <= 300:
                filtered_words.append(word)
            else:
                break

        # Find the largest font size among the filtered words, and round it down
        largest_font_size = max([(word['bottom'] - word['top']) for word in filtered_words])
        largest_font_size = round(largest_font_size, 0)  # Round down to nearest whole number

        # Gather all words with this rounded font size, excluding the word "Abstract"
        title_words = [word['text'] for word in filtered_words if round((word['bottom'] - word['top']), 0) == largest_font_size and word['text'].lower() != 'abstract']

        return ' '.join(title_words)

# Method 1 from first code block
def extract_title_method1(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        words = first_page.extract_words()

        # Filter out words that appear after the first 300 characters
        text_so_far = ''
        filtered_words = []
        for word in words:
            text_so_far += word['text']
            if len(text_so_far) <= 300:
                filtered_words.append(word)
            else:
                break

        largest_font_size = max([(word['bottom'] - word['top']) for word in filtered_words])

        title_candidate_words = [word for word in filtered_words if (word['bottom'] - word['top']) == largest_font_size and word['text'].lower() != 'abstract']

        title_sequences = []
        current_sequence = []
        for i, word in enumerate(title_candidate_words):
            if i > 0 and (word['x0'] - title_candidate_words[i-1]['x1'] > 10 or word['top'] - title_candidate_words[i-1]['bottom'] > 10):
                title_sequences.append(current_sequence)
                current_sequence = []
            current_sequence.append(word['text'])
        if current_sequence:
            title_sequences.append(current_sequence)

        title = ' '.join(max(title_sequences, key=len) if title_sequences else [])
        return title

# Method 2 using pdfplumber
def extract_title_with_pdfplumber(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        words = first_page.extract_words()

        text_so_far = ''
        filtered_words = []
        for word in words:
            text_so_far += word['text']
            if len(text_so_far) <= 300:
                filtered_words.append(word)
            else:
                break

        largest_font_size = max([(word['bottom'] - word['top']) for word in filtered_words])

        title_words = [word['text'] for word in filtered_words if (word['bottom'] - word['top']) == largest_font_size]
        return ' '.join(title_words)

# Extract Abstract
def extract_abstract(text):
    abstract_start_pattern = re.compile(r'(?i)abstract')
    introduction_pattern = re.compile(r'(?i)1\.\s*[_]?[lntrnd_uction]+')
    end_patterns = [introduction_pattern, re.compile(r'(?i)All rights'), re.compile(r'(?i)c/circlecopy'), re.compile(r'(?i)keywords'), re.compile(r'(?i)\(c\)')]

    abstract_start = abstract_start_pattern.search(text)
    abstract_ends = [pattern.search(text, (abstract_start.start() if abstract_start else 0)) for pattern in end_patterns]
    abstract_ends = [match.start() for match in abstract_ends if match]

    if not abstract_ends:
        abstract_end_keywords = ['All rights', 'c/circlecopy', '1. Introduction', 'Introduction', '1. _lntrnd_uction', '(c)', 'keywords', 'Keywords']
        abstract_ends = [text.lower().find(keyword.lower(), abstract_start.start()) for keyword in abstract_end_keywords if text.lower().find(keyword.lower(), abstract_start.start()) != -1]

    abstract_end = min(abstract_ends) if abstract_ends else -1
    abstract = re.sub(r'(?i)abstract', '', text[abstract_start.start():abstract_end]).strip() if abstract_start and abstract_end != -1 else None
    return abstract

# Additional Title Extraction
def extract_basic_title(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        words = first_page.extract_words()

        # Get the largest font size
        largest_font_size = max([(word['bottom'] - word['top']) for word in words])

        # Extract title words based on largest font size only
        title_words = [word['text'] for word in words if (word['bottom'] - word['top']) == largest_font_size]

        # Join the title words
        return ' '.join(title_words)

#Main for text preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')