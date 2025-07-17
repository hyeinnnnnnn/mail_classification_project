import sys
import pandas as pd # pandas for CSV reading
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql # Still needed for initial keyword table loading
import torch
from collections import defaultdict

# --- Configuration ---
CSV_FILE_PATH = "C:\\Users\\KOPO\\Documents\\UiPath\\newcrowring\\finance_news.xlsx" # Your CSV/Excel file path
TEXT_COLUMN_NAME = "detail" # Name of the column in your CSV/Excel that contains the text to classify

# Model and Morphological Analyzer Initialization
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()

# DB Connection (for loading keyword tables only)
try:
    conn = pymysql.connect(
        host='10.50.131.18',
        port=3306,
        user='user',
        password='user1234',
        database='maildb',
        charset='utf8'
    )
    cursor = conn.cursor()
    print("DB connection successful for keyword loading.")

except pymysql.Error as e:
    print(f"DB connection error for keyword loading: {e}")
    sys.exit(1)

# Keyword tables list (categories)
keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']

# --- Helper Functions ---

# Extract nouns only (NNG: general noun, NNP: proper noun)
def extract_keywords(text):
    if not isinstance(text, str): # Handle non-string input (e.g., NaN from Excel)
        return []
    tokens = kiwi.tokenize(text)
    return [token.form for token in tokens if token.tag in ("NNG", "NNP") and token.form.isalpha() and len(token.form) > 1] # Exclude single-character words

# Function to get keywords from a specific table
def get_keywords_from_table(table_name):
    try:
        cursor.execute(f"SELECT word FROM {table_name}")
        return [row[0] for row in cursor.fetchall()]
    except pymysql.Error as e:
        print(f"Error fetching keywords from {table_name}: {e}")
        return []

# Function to load text data from CSV/Excel
def load_text_from_csv(file_path, column_name):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please provide a .csv, .xlsx, or .xls file.")
            sys.exit(1)

        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the file.")
            print(f"Available columns are: {df.columns.tolist()}")
            sys.exit(1)

        # Convert to list and handle potential NaN values (empty cells)
        text_list = df[column_name].astype(str).tolist()
        return text_list
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from file: {e}")
        sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    # Command-line argument for target_user_id (still kept for potential identification/logging)
    if len(sys.argv) < 2:
        print("Usage: python mail_classifier_csv.py <target_identifier_for_output>")
        print("Example: python mail_classifier_csv.py user123_data")
        sys.exit(1)

    target_identifier = sys.argv[1]
    print(f"\n--- Classifying text data from '{CSV_FILE_PATH}' for identifier '{target_identifier}' ---")

    # Load text data from the specified CSV/Excel file
    texts_to_classify = load_text_from_csv(CSV_FILE_PATH, TEXT_COLUMN_NAME)

    if not texts_to_classify:
        print("No text data found to classify. Exiting.")
        conn.close()
        sys.exit(0)

    # Load and embed all keywords from DB tables (cached for efficiency)
    all_table_keywords = {}
    all_keyword_embeddings = {}

    for table_name in keyword_tables:
        keywords = get_keywords_from_table(table_name)
        all_table_keywords[table_name] = keywords
        if keywords:
            all_keyword_embeddings[table_name] = model.encode(keywords, convert_to_tensor=True)
        else:
            all_keyword_embeddings[table_name] = torch.tensor([])
        print(f"Loaded {len(keywords)} keywords for '{table_name}'.")

    # Classification related thresholds
    TOP_K_KEYWORDS = 100  # Use the top K most similar keywords for category voting
    MIN_SIMILARITY_FOR_CANDIDATE = 0.35  # Minimum similarity for a keyword to be considered a candidate
    FINAL_CLASSIFICATION_THRESHOLD = 0.6  # Minimum overall similarity to be classified into a specific category

    # --- Main Classification Loop ---
    for text_idx, text_content in enumerate(texts_to_classify):
        print(f"\n--- {text_idx + 1}/{len(texts_to_classify)} Classifying Text ---")
        display_text = text_content if len(text_content) < 70 else text_content[:67] + "..."
        print(f"  Input Text: '{display_text}'")

        tokens = extract_keywords(text_content)
        if not tokens:
            classified_category = "Unclassified (No_Keywords_Extracted)"
            print(f"  Result: {classified_category}")
            continue

        text_token_embeddings = model.encode(tokens, convert_to_tensor=True)

        all_candidate_similarities = []

        # Calculate similarities between each token from the input text and all category keywords
        for table_name in keyword_tables:
            keywords_in_table = all_table_keywords[table_name]
            embeddings_in_table = all_keyword_embeddings[table_name]

            if not embeddings_in_table.numel(): # Skip if no keywords in table
                continue

            # Compute cosine similarity matrix
            similarity_matrix = util.pytorch_cos_sim(text_token_embeddings, embeddings_in_table)

            for i, text_token in enumerate(tokens):
                for j, keyword_in_table in enumerate(keywords_in_table):
                    sim_score = similarity_matrix[i][j].item()
                    if sim_score >= MIN_SIMILARITY_FOR_CANDIDATE:
                        all_candidate_similarities.append({
                            'text_token': text_token,
                            'keyword_in_table': keyword_in_table,
                            'similarity': sim_score,
                            'category': table_name.replace('_keywords', '') # Clean category name
                        })

        # Sort by similarity in descending order and select top K candidates
        all_candidate_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_k_candidates = all_candidate_similarities[:TOP_K_KEYWORDS]

        if not top_k_candidates:
            classified_category = "Unclassified (No_High_Sim_Candidates)"
            print(f"  Result: {classified_category}")
            continue

        # Aggregate votes and average similarities per category from top K candidates
        category_votes = defaultdict(int)
        category_sim_sums = defaultdict(float)
        category_sim_counts = defaultdict(int)
        max_overall_similarity = 0.0 # Track the single highest similarity score among top K

        for item in top_k_candidates:
            category_votes[item['category']] += 1
            category_sim_sums[item['category']] += item['similarity']
            category_sim_counts[item['category']] += 1
            if item['similarity'] > max_overall_similarity:
                max_overall_similarity = item['similarity']

        if not category_votes: # Should not happen if top_k_candidates is not empty, but for safety
            classified_category = "Unclassified (No_Votes_After_Top_K)"
            print(f"  Result: {classified_category}")
            continue

        # Determine the category with the most votes
        max_votes = 0
        for cat, votes in category_votes.items():
            if votes > max_votes:
                max_votes = votes

        # Find all categories with the maximum votes
        candidate_categories = [cat for cat, votes in category_votes.items() if votes == max_votes]

        # If there's a tie in votes, compare average similarities
        if len(candidate_categories) > 1:
            best_candidate_category = candidate_categories[0]
            highest_avg_sim_among_candidates = category_sim_sums[best_candidate_category] / category_sim_counts[
                best_candidate_category]

            for i in range(1, len(candidate_categories)):
                cat = candidate_categories[i]
                current_avg_sim = category_sim_sums[cat] / category_sim_counts[cat]
                if current_avg_sim > highest_avg_sim_among_candidates:
                    highest_avg_sim_among_candidates = current_avg_sim
                    best_candidate_category = cat
            classified_category = best_candidate_category
        else:
            classified_category = candidate_categories[0]

        # Final classification based on the overall highest similarity threshold
        if max_overall_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
            print(f"  Result: {classified_category} (Max Sim: {max_overall_similarity:.2f})")
        else:
            classified_category = "Unclassified" # If overall similarity is too low
            print(f"  Result: {classified_category} (Max Sim in Top {TOP_K_KEYWORDS} is {max_overall_similarity:.2f}, below threshold {FINAL_CLASSIFICATION_THRESHOLD:.2f})")


# --- DB Connection Closure ---
cursor.close()
conn.close()
print(f"\n--- Classification for '{target_identifier}' completed ---")