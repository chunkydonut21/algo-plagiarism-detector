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
def calculate_overall_similarity(results):
    """Calculate the average n-gram similarity percentage across all documents."""
    similarities = [result['ngram_similarity'] for result in results]
    return sum(similarities) / len(similarities) if similarities else 0

def plagiarism_status(overall_rate):
    """Determine the plagiarism status and color based on the rate."""
    if overall_rate < 40:
        return "OK", "green"
    elif 40 <= overall_rate < 75:
        return "Warning", "orange"
    else:
        return "Plagiarized", "red"
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
def visualize_results(results, repository_files):
    overall_similarity = calculate_overall_similarity(results)
    overall_status, overall_color = plagiarism_status(overall_similarity)

    # Display overall plagiarism rate
    st.header("Plagiarism Report")

    # Visualize individual document comparisons
    for result, repo_file in zip(results, repository_files):
        repo_name = repo_file[0]
        file_status, file_color = plagiarism_status(result['ngram_similarity'])

        st.subheader(f"Repository File: {repo_name}")
        st.markdown(
            f"<h4 style='color:{file_color};'>Plagiarism Rate: {result['ngram_similarity']:.2f}% ({file_status})</h4>",
            unsafe_allow_html=True
        )

        st.write(f"Edit Distance: {result['edit_distance']}")
        st.write(f"N-Gram Similarity: {result['ngram_similarity']:.2f}%")
        st.write(f"Longest Common Substring: {result['longest_common_substring']}")
        st.write(f"Common Token Count: {result['common_count']}")
        st.write(f"Common Tokens: {', '.join(result['common_tokens'])}")

        # Bar chart for metrics
        metrics = [result['edit_distance'], result['ngram_similarity'], result['common_count']]
        labels = ['Edit Distance (lower is better)', 'N-Gram Similarity (%)', 'Common Token Count']

        fig, ax = plt.subplots()
        ax.barh(labels, metrics, color=['red', 'green', 'blue'])
        ax.set_xlabel("Values")
        ax.set_title(f"Metrics for {repo_name}")
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

# Updated plagiarism_ui function
def plagiarism_ui():
    st.title("Plagiarism Detection System")

    st.sidebar.header("Input Options")
    st.sidebar.subheader("Input 1: Submission File")

    # Single Submission File
    submission_file = st.sidebar.file_uploader("Upload a Submission File", type=["txt", "pdf"])

    submission_text = ""
    if submission_file:
        submission_text = read_file(submission_file)

    st.sidebar.subheader("Input 2: Repository Files")

    # Multiple Repository Files
    repository_files = st.sidebar.file_uploader("Upload Repository Files (Multiple Allowed)", type=["txt", "pdf"], accept_multiple_files=True)

    repository_contents = []
    if repository_files:
        for file in repository_files:
            repository_contents.append((file.name, read_file(file)))

    if st.sidebar.button("Analyze"):
        if submission_text and repository_contents:
            st.header("Results")
            results = []

            for repo_name, repo_text in repository_contents:
                # Check plagiarism for each repository file
                result = detect_plagiarism(submission_text, [repo_text])[0]  # Single repository document
                results.append(result)

            # Visualize plagiarism results
            visualize_results(results, repository_contents)
        else:
            st.error("Please provide both a submission file and at least one repository file.")

# Example Usage
if __name__ == "__main__":
    plagiarism_ui()
