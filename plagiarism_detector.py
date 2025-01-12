import re
from collections import defaultdict
import streamlit as st
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

# 1. Preprocessing and Normalization
def preprocess_text(text):
    # Remove extra whitespaces and special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# 2. Custom Edit Distance (Levenshtein Distance)
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

# 3. Custom Longest Common Substring (LCS)
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    lcs_length = 0
    end_idx = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > lcs_length:
                    lcs_length = dp[i][j]
                    end_idx = i

    return s1[end_idx - lcs_length:end_idx]

# 4. Custom N-Gram Similarity
def ngram_similarity(s1, s2, n=3):
    def generate_ngrams(text, n):
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    ngrams1 = generate_ngrams(s1, n)
    ngrams2 = generate_ngrams(s2, n)

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) * 100 if union else 0

# 5. Hash Table for Tokenized Text Comparison
def tokenize_and_hash(text):
    tokens = text.split()
    hash_table = defaultdict(list)
    for i, token in enumerate(tokens):
        hash_table[token].append(i)
    return hash_table

def compare_hash_tables(hash_table1, hash_table2):
    common_tokens = set(hash_table1.keys()).intersection(set(hash_table2.keys()))
    return len(common_tokens), common_tokens

# 6. Plagiarism Detection Engine
def detect_plagiarism(submission, repository):
    submission = preprocess_text(submission)
    repository = [preprocess_text(doc) for doc in repository]

    results = []
    for doc_index, doc in enumerate(repository):
        edit_dist = edit_distance(submission, doc)
        lcs = longest_common_substring(submission, doc)
        ngram_sim = ngram_similarity(submission, doc)
        submission_hash = tokenize_and_hash(submission)
        doc_hash = tokenize_and_hash(doc)
        common_count, common_tokens = compare_hash_tables(submission_hash, doc_hash)

        results.append({
            "doc_index": doc_index,
            "edit_distance": edit_dist,
            "longest_common_substring": lcs,
            "ngram_similarity": ngram_sim,
            "common_count": common_count,
            "common_tokens": common_tokens
        })
    return results

# 7. Enhanced Report Visualization
def visualize_results(results, repository):
    for result in results:
        st.subheader(f"Document {result['doc_index']}")

        st.write(f"Edit Distance: {result['edit_distance']}")
        st.write(f"N-Gram Similarity: {result['ngram_similarity']:.2f}%")
        st.write(f"Longest Common Substring: {result['longest_common_substring']}")
        st.write(f"Common Token Count: {result['common_count']}")
        st.write(f"Common Tokens: {', '.join(result['common_tokens'])}")

        # Add a bar chart to visualize similarity metrics
        metrics = [result['edit_distance'], result['ngram_similarity'], result['common_count']]
        labels = ['Edit Distance (lower is better)', 'N-Gram Similarity (%)', 'Common Token Count']

        fig, ax = plt.subplots()
        ax.barh(labels, metrics, color=['red', 'green', 'blue'])
        ax.set_xlabel("Values")
        ax.set_title(f"Metrics for Document {result['doc_index']}")
        st.pyplot(fig)

# 8. Streamlit UI for Plagiarism Detection
def read_file(file):
    if file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    elif file.name.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return ""

def plagiarism_ui():
    st.title("Plagiarism Detection System")

    st.sidebar.header("Input Options")
    submission_text = st.sidebar.text_area("Enter the Submission Text")
    submission_file = st.sidebar.file_uploader("Or Upload a Submission File", type=["txt", "pdf"])

    repository_files = st.sidebar.file_uploader("Upload Repository Files (Multiple)", type=["txt", "pdf"], accept_multiple_files=True)
    repository_texts = st.sidebar.text_area("Or Enter Repository Texts (Separate by New Line)").split('\n')

    if submission_file:
        submission_text = read_file(submission_file)

    repository_contents = []
    if repository_files:
        for file in repository_files:
            repository_contents.append(read_file(file))

    if repository_texts:
        repository_contents.extend(repository_texts)

    if st.sidebar.button("Analyze"):
        if submission_text and repository_contents:
            results = detect_plagiarism(submission_text, repository_contents)
            st.header("Results")
            visualize_results(results, repository_contents)
        else:
            st.error("Please provide both submission text and repository files/texts.")

# Example Usage
if __name__ == "__main__":
    plagiarism_ui()
