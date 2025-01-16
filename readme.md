# Plagiarism Detection System

## Overview
This project is a **Plagiarism Detection System** built with Python, leveraging algorithms for text preprocessing, similarity analysis, and visualization. The tool provides an intuitive user interface using **Streamlit**, enabling users to upload or input texts and repository documents for comparison. The system highlights similarities between texts using advanced metrics and visualizations.

You can access the deployed application here:  
👉 [Plagiarism Detection System](https://plagiarism-detector-ef1n.onrender.com/)

---

## Features
1. **Text Preprocessing:**
   - Normalizes text by removing special characters, extra whitespaces, and converting to lowercase.

2. **Similarity Metrics:**
   - **Edit Distance (Levenshtein Distance):** Calculates the minimum changes required to transform one string into another.
   - **Longest Common Substring (LCS):** Finds the longest substring shared between two texts.
   - **N-Gram Similarity:** Compares texts by generating and analyzing n-grams.
   - **Token Hash Comparison:** Identifies and counts common tokens between texts using hash tables.

3. **Plagiarism Detection:**
   - Compares a submission against a repository of documents.
   - Provides detailed metrics and highlights common tokens.

4. **Visualization:**
   - Displays results using bar charts for metrics like edit distance, n-gram similarity, and common token count.

5. **File Support:**
   - Accepts `.txt` and `.pdf` file formats.
   - Allows multiple repository files to be uploaded.

6. **Interactive UI:**
   - Built with Streamlit for a user-friendly experience.
   - Supports both text input and file uploads.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/chunkydonut21/algo-plagiarism-detector
   cd plagiarism-detection-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Launch the application:
   ```bash
   streamlit run app.py
   ```
2. Navigate to the app in your browser.
3. Enter or upload:
   - **Submission Text/File:** The text or document to analyze.
   - **Repository Files/Texts:** The reference documents for comparison.
4. Click **Analyze** to view the results:
   - Metrics for each repository document.
   - Bar charts visualizing similarity metrics.
   - Detailed comparison of common tokens.

---

## Code Structure
- **Preprocessing:**
  - `preprocess_text(text)`: Cleans and normalizes input text.
- **Similarity Algorithms:**
  - `edit_distance(s1, s2)`: Computes the Levenshtein distance.
  - `longest_common_substring(s1, s2)`: Finds the longest shared substring.
  - `ngram_similarity(s1, s2, n)`: Calculates n-gram-based similarity.
  - `tokenize_and_hash(text)`: Tokenizes and hashes text for comparison.
  - `compare_hash_tables(hash_table1, hash_table2)`: Identifies and counts common tokens.
- **Plagiarism Detection:**
  - `detect_plagiarism(submission, repository)`: Compares a submission against repository documents.
- **Visualization:**
  - `visualize_results(results, repository)`: Visualizes similarity metrics using Streamlit and Matplotlib.
- **UI:**
  - `plagiarism_ui()`: Handles the user interface for input, file upload, and results display.

---

## Requirements
- Python 3.8 or higher
- Libraries:
  - `streamlit`
  - `matplotlib`
  - `PyPDF2`

---

## Future Enhancements
- Add support for other file formats (e.g., `.docx`).
- Incorporate semantic similarity using machine learning models.
- Add functionality for large-scale document repositories.
- Improve UI with advanced filtering options.

---
