import re
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from suffix_trees import STree
import streamlit as st
import numpy as np

# 1. Preprocessing and Normalization
def preprocess_text(text):
    # Remove extra whitespaces and special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# 2. Rabin-Karp Algorithm for String Matching
def rabin_karp(pattern, text, prime=101):
    m = len(pattern)
    n = len(text)
    pattern_hash = 0
    text_hash = 0
    h = 1

    for i in range(m - 1):
        h = (h * 256) % prime

    for i in range(m):
        pattern_hash = (256 * pattern_hash + ord(pattern[i])) % prime
        text_hash = (256 * text_hash + ord(text[i])) % prime

    matches = []
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i + m] == pattern:
                matches.append(i)

        if i < n - m:
            text_hash = (256 * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime

    return matches

# 3. Text Similarity Metrics
def compute_similarity(text1, text2):
    ratio = SequenceMatcher(None, text1, text2).ratio() * 100
    edit_distance = levenshtein_distance(text1, text2)
    return ratio, edit_distance

# 4. Tokenization and N-Gram Similarity
def compute_ngram_similarity(text1, text2, n=3):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    vectors = vectorizer.fit_transform([text1, text2]).toarray()
    similarity = cosine_similarity(vectors)[0, 1] * 100
    return similarity

# 5. Suffix Tree for String Matching
def suffix_tree_matching(submission, document):
    stree = STree.STree([submission, document])
    matches = stree.lcs()  # Longest common substring
    return matches

# 6. Hash Table for Tokenized Text Comparison
def tokenize_and_hash(text):
    tokens = text.split()
    hash_table = defaultdict(list)
    for i, token in enumerate(tokens):
        hash_table[token].append(i)
    return hash_table

def compare_hash_tables(hash_table1, hash_table2):
    common_tokens = set(hash_table1.keys()).intersection(set(hash_table2.keys()))
    return len(common_tokens), common_tokens

# 7. Plagiarism Detection Engine
def detect_plagiarism(submission, repository):
    submission = preprocess_text(submission)
    repository = [preprocess_text(doc) for doc in repository]

    results = []
    for doc_index, doc in enumerate(repository):
        ratio, edit_distance = compute_similarity(submission, doc)
        ngram_similarity = compute_ngram_similarity(submission, doc)
        lcs = suffix_tree_matching(submission, doc)
        submission_hash = tokenize_and_hash(submission)
        doc_hash = tokenize_and_hash(doc)
        common_count, common_tokens = compare_hash_tables(submission_hash, doc_hash)

        results.append({
            "doc_index": doc_index,
            "similarity_ratio": ratio,
            "edit_distance": edit_distance,
            "ngram_similarity": ngram_similarity,
            "longest_common_substring": lcs,
            "common_count": common_count,
            "common_tokens": common_tokens
        })
    return results

# 8. Report Generation
def generate_report(results, repository):
    reports = []
    for result in results:
        report = {
            "Document": result['doc_index'],
            "Similarity Ratio": f"{result['similarity_ratio']:.2f}%",
            "Edit Distance": result['edit_distance'],
            "N-Gram Similarity": f"{result['ngram_similarity']:.2f}%",
            "Longest Common Substring": result['longest_common_substring'],
            "Common Token Count": result['common_count'],
            "Common Tokens": result['common_tokens']
        }
        reports.append(report)
    return reports

# 9. Streamlit UI for Plagiarism Detection
def plagiarism_ui():
    st.title("Plagiarism Detection System")

    st.sidebar.header("Upload Files")
    submission_file = st.sidebar.text_area("Enter the Submission Text")
    repository_files = st.sidebar.text_area("Enter Repository Texts (Separate by New Line)").split('\n')

    if st.sidebar.button("Analyze"):
        results = detect_plagiarism(submission_file, repository_files)
        reports = generate_report(results, repository_files)

        st.header("Results")
        for report in reports:
            st.subheader(f"Document {report['Document']}")
            st.write(f"Similarity Ratio: {report['Similarity Ratio']}")
            st.write(f"Edit Distance: {report['Edit Distance']}")
            st.write(f"N-Gram Similarity: {report['N-Gram Similarity']}")
            st.write(f"Longest Common Substring: {report['Longest Common Substring']}")
            st.write(f"Common Token Count: {report['Common Token Count']}")
            st.write(f"Common Tokens: {', '.join(report['Common Tokens'])}")

# Example Usage
if __name__ == "__main__":
    plagiarism_ui()
