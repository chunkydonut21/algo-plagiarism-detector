import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import streamlit as st
import matplotlib.pyplot as plt


# 1. Preprocessing and Normalization
def preprocess_text(text: str) -> str:
    """
    Removes extra whitespaces, special characters, and converts text to lowercase.
    """
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    return text.lower().strip()


# 2. Custom Edit Distance (Levenshtein Distance)
def edit_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein distance between two strings s1 and s2.
    """
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
def longest_common_substring(s1: str, s2: str) -> str:
    """
    Returns the longest common substring between s1 and s2.
    """
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
def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Computes the n-gram similarity percentage between s1 and s2.
    """
    def generate_ngrams(text: str, n: int) -> Set[str]:
        if len(text) < n:
            return set()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    ngrams1 = generate_ngrams(s1, n)
    ngrams2 = generate_ngrams(s2, n)

    union = ngrams1 | ngrams2
    intersection = ngrams1 & ngrams2

    if not union:
        return 0.0
    return (len(intersection) / len(union)) * 100


# 5. Hash Table for Tokenized Text Comparison
def tokenize_and_hash(text: str) -> Dict[str, List[int]]:
    """
    Splits text into tokens and stores their positions in a hash table.
    """
    tokens = text.split()
    hash_table: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(tokens):
        hash_table[token].append(i)
    return hash_table


def compare_hash_tables(
    hash_table1: Dict[str, List[int]],
    hash_table2: Dict[str, List[int]]
) -> Tuple[int, Set[str]]:
    """
    Returns the number of common tokens and the set of those common tokens.
    """
    common_tokens = set(hash_table1.keys()).intersection(hash_table2.keys())
    return len(common_tokens), common_tokens


# 6. Plagiarism Detection Engine (with Plagiarism Rate)
def detect_plagiarism(submission: str, repository: List[str]) -> List[Dict[str, Any]]:
    """
    Detects plagiarism indicators by comparing a submission against a list of repository documents.
    Also calculates a plagiarism rate for each document.
    """
    if not submission.strip():
        return []

    submission = preprocess_text(submission)

    # Preprocess repository documents
    preprocessed_repo = [preprocess_text(doc) for doc in repository if doc.strip()]

    results = []
    for doc_index, doc in enumerate(preprocessed_repo):
        edit_dist = edit_distance(submission, doc)
        lcs = longest_common_substring(submission, doc)
        ngram_sim = ngram_similarity(submission, doc)

        submission_hash = tokenize_and_hash(submission)
        doc_hash = tokenize_and_hash(doc)

        common_count, common_tokens = compare_hash_tables(submission_hash, doc_hash)

        # Calculate plagiarism rate as a weighted average of similarities
        plagiarism_rate = (
            (ngram_sim * 0.5) + (len(common_tokens) / max(len(submission.split()), 1) * 50)
        )

        results.append({
            "doc_index": doc_index,
            "edit_distance": edit_dist,
            "longest_common_substring": lcs,
            "ngram_similarity": ngram_sim,
            "common_count": common_count,
            "common_tokens": common_tokens,
            "plagiarism_rate": plagiarism_rate,
        })
    return results


# 7. Enhanced Report Visualization (with Overall Plagiarism Rate)
def visualize_results(results: List[Dict[str, Any]], repository: List[str]) -> None:
    """
    Displays the results of the plagiarism detection using Streamlit and matplotlib,
    including an overall plagiarism rate.
    """
    if not results:
        st.warning("No valid results to display.")
        return

    # Calculate overall plagiarism rate
    overall_rate = sum(result['plagiarism_rate'] for result in results) / len(results)
    st.header(f"Overall Plagiarism Rate: {overall_rate:.2f}%")

    for result in results:
        st.subheader(f"Document {result['doc_index']}")
        st.write(f"Edit Distance: {result['edit_distance']}")
        st.write(f"N-Gram Similarity: {result['ngram_similarity']:.2f}%")
        st.write(f"Longest Common Substring: '{result['longest_common_substring']}'")
        st.write(f"Common Token Count: {result['common_count']}")
        st.write(f"Common Tokens: {', '.join(result['common_tokens'])}")
        st.write(f"Plagiarism Rate: {result['plagiarism_rate']:.2f}%")

        # Bar chart visualization for this document
        metrics = [
            result['edit_distance'], 
            result['ngram_similarity'], 
            result['common_count'], 
            result['plagiarism_rate']
        ]
        labels = [
            "Edit Distance (lower is better)", 
            "N-Gram Similarity (%)", 
            "Common Token Count", 
            "Plagiarism Rate (%)"
        ]

        fig, ax = plt.subplots()
        ax.barh(labels, metrics, color=["red", "green", "blue", "purple"])
        ax.set_xlabel("Values")
        ax.set_title(f"Metrics for Document {result['doc_index']}")
        st.pyplot(fig)


# 8. Streamlit UI for Plagiarism Detection
def plagiarism_ui() -> None:
    """
    Streamlit UI that allows the user to input a submission text and repository documents,
    and then analyzes them for potential plagiarism.
    """
    st.title("Plagiarism Detection System")

    st.sidebar.header("Submission & Repository Inputs")
    submission_text = st.sidebar.text_area("Enter the Submission Text", height=150)
    repo_texts = st.sidebar.text_area(
        "Enter Repository Texts (Separate by New Line)",
        height=150
    ).split('\n')

    if st.sidebar.button("Analyze"):
        if not submission_text.strip():
            st.error("Submission text is empty. Please provide valid text.")
            return

        results = detect_plagiarism(submission_text, repo_texts)

        if not results:
            st.warning("No valid repository documents or empty submission.")
        else:
            st.header("Results")
            visualize_results(results, repo_texts)


# Example Usage
if __name__ == "__main__":
    plagiarism_ui()
