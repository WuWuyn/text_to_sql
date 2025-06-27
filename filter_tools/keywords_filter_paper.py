import pandas as pd
from fuzzywuzzy import fuzz
import re
from joblib import Parallel, delayed
import os

class KeywordFilter:
    def __init__(self):
        """Initialize the KeywordFilter class without NLTK dependency."""
        # Define default keyword groups
        self.text_to_sql_keywords = [
            "Software Testing", "Testing", "Test Automation", "Test Case Generation",
            "Test Script Generation", "Test Data Generation", "Test Oracle Generation", "Test Repair"
        ]
        self.security_keywords = [
            "Functional Testing", "System Testing", "End-to-End Testing", "GUI Testing", 
            "UI Testing", "Web Testing", "Mobile Testing", "Agent", "AI Agent", 
            "Autonomous Agent", "Prompt Engineering", "Chain-of-Thought", "Retrieval-Augmented Generation"
        ]
        self.llm_keywords = ["llm", "large language model"]
        print("KeywordFilter initialized. Using built-in sentence tokenizer and n-gram generator.")

    def tokenize_sentences(self, text):
        """
        Simple sentence tokenizer using regex.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of sentences
        """
        if not isinstance(text, str):
            return []
        # Simple regex to split on sentence boundaries (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def set_keywords(self, t2sql_keywords=None, security_keywords=None, llm_keywords=None):
        """
        Set custom keywords for each category.
        
        Args:
            t2sql_keywords (list): List of text-to-SQL related keywords
            security_keywords (list): List of security related keywords
            llm_keywords (list): List of LLM related keywords
        """
        if t2sql_keywords:
            self.text_to_sql_keywords = t2sql_keywords
        if security_keywords:
            self.security_keywords = security_keywords
        if llm_keywords:
            self.llm_keywords = llm_keywords
    
    def preprocess_text(self, text):
        """
        Preprocess the text: convert to lowercase, replace punctuation with space, and reduce multiple spaces to one.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        text = re.sub(r'\s+', ' ', text)  # Reduce multiple spaces to one
        text = text.strip()  # Remove leading/trailing spaces
        return text
    
    def generate_ngrams(self, text, n):
        """
        Generate n-grams from the text using pure Python.
        
        Args:
            text (str): Text to generate n-grams from
            n (int): Size of n-grams
            
        Returns:
            list: List of n-grams as strings
        """
        words = text.split()
        # Ensure there are enough words to create an n-gram
        if len(words) < n:
            return []
        
        # Use zip to create n-grams efficiently
        ngrams_tuples = zip(*(words[i:] for i in range(n)))
        return [' '.join(gram) for gram in ngrams_tuples]
    
    def check_group(self, sentences, keywords, threshold=90):
        """
        Check if the group of keywords is related to the text.
        
        Args:
            sentences (list): List of sentences
            keywords (list): List of keywords to check
            threshold (int): Minimum similarity threshold (0-100)
            
        Returns:
            bool: True if any keyword matches above threshold, False otherwise
        """
        all_ngrams = []
        for sentence in sentences:
            preprocessed_sentence = self.preprocess_text(sentence)
            for n in range(1, 5):  # From 1-gram to 4-gram
                all_ngrams.extend(self.generate_ngrams(preprocessed_sentence, n))
        
        # Preprocess keywords once to avoid redundant work
        processed_keywords = [self.preprocess_text(k) for k in keywords]

        for keyword in processed_keywords:
            for ngram in all_ngrams:
                if fuzz.WRatio(keyword, ngram) > threshold:
                    return True
            # Also check against the full sentences for broader context matches
            for sentence in sentences:
                if fuzz.WRatio(keyword, self.preprocess_text(sentence)) > threshold:
                    return True
        return False
    
    def classify_abstract(self, abstract):
        """
        Classify the abstract and return a list of True/False for each keyword group.
        
        Args:
            abstract (str): Abstract text to classify
            
        Returns:
            list: List of boolean values [functional_testing_result, functional_result, llm_result]
        """
        # We don't need to preprocess here, as check_group handles it internally
        sentences = self.tokenize_sentences(abstract)
        
        functional_testing_result = self.check_group(sentences, self.text_to_sql_keywords)
        functional_result = self.check_group(sentences, self.security_keywords)
        llm_result = self.check_group(sentences, self.llm_keywords)
        
        return [functional_testing_result, functional_result, llm_result]
    
    def process_csv(self, input_csv, output_csv):
        """
        Read a CSV file, classify each abstract with parallel processing,
        and add 3 new boolean columns: functional_testing, functional, llm.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str, optional): Path to save output CSV file
            
        Returns:
            pd.DataFrame: Updated DataFrame with new boolean columns
        """
        df = pd.read_csv(input_csv)
        
        if 'abstract' not in df.columns:
            raise ValueError("CSV file must contain an 'abstract' column.")
        
        df['abstract'] = df['abstract'].fillna('').astype(str) # Handle missing abstracts
        classifications = Parallel(n_jobs=-1)(delayed(self.classify_abstract)(abstract) for abstract in df['abstract'])
        
        df['t2sql'] = [result[0] for result in classifications]  # Keep old column name for compatibility
        df['security'] = [result[1] for result in classifications]  # Keep old column name for compatibility 
        df['llm'] = [result[2] for result in classifications]
        
        # Add new column names
        df['functional_testing'] = df['t2sql']
        df['functional'] = df['security']
        
        if 'submitted' in df.columns:
            df['submitted'] = pd.to_datetime(df['submitted'], errors='coerce')
            df = df.sort_values(by='submitted', ascending=False)

        if output_csv:
            # Ensure directory exists
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_csv, index=False)
        
        return df
    
    # The generate_keyword_summaries and run_pipeline methods remain the same
    # as they do not depend on NLTK. For completeness, they are included below.

    def generate_keyword_summaries(self, df, output_dir):
        """
        Generate summary CSV files for different keyword combinations.
        
        Args:
            df (pd.DataFrame): DataFrame with t2sql, security, and llm columns
            output_dir (str): Directory to save summary CSV files
            
        Returns:
            dict: Dictionary containing the summary DataFrames
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Papers with all three keyword types
        all_true_df = df[(df['t2sql'] == True) & (df['llm'] == True) & (df['security'] == True)].copy()
        if 'submitted' in all_true_df.columns:
            all_true_df['submitted'] = pd.to_datetime(all_true_df['submitted'])
            all_true_df = all_true_df.sort_values(by='submitted', ascending=False)
        all_true_df.to_csv(f"{output_dir}/all_true.csv", index=False)
        
        # 2. Papers with functional_testing and llm keywords
        functional_testing_llm_true_df = df[(df['t2sql'] == True) & (df['llm'] == True)].copy()
        if 'submitted' in functional_testing_llm_true_df.columns:
            functional_testing_llm_true_df['submitted'] = pd.to_datetime(functional_testing_llm_true_df['submitted'])
            functional_testing_llm_true_df = functional_testing_llm_true_df.sort_values(by='submitted', ascending=False)
        functional_testing_llm_true_df.to_csv(f"{output_dir}/functional_testing_llm_true.csv", index=False)
        
        # 3. Papers with functional_testing and functional keywords
        functional_testing_functional_true_df = df[(df['t2sql'] == True) & (df['security'] == True)].copy()
        if 'submitted' in functional_testing_functional_true_df.columns:
            functional_testing_functional_true_df['submitted'] = pd.to_datetime(functional_testing_functional_true_df['submitted'])
            functional_testing_functional_true_df = functional_testing_functional_true_df.sort_values(by='submitted', ascending=False)
        functional_testing_functional_true_df.to_csv(f"{output_dir}/functional_testing_functional_true.csv", index=False)
        
        # 4. Merge all results and remove duplicates
        merged_df = pd.concat([all_true_df, functional_testing_llm_true_df, functional_testing_functional_true_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates()
        merged_df.to_csv(f"{output_dir}/merged_keyword_results.csv", index=False)
        
        return {
            'all_true': all_true_df,
            'functional_testing_llm_true': functional_testing_llm_true_df,
            'functional_testing_functional_true': functional_testing_functional_true_df,
            'merged': merged_df
        }
    
    def run_pipeline(self, input_csv, output_dir):
        """
        Run the complete pipeline: process CSV and generate summaries.
        
        Args:
            input_csv (str): Path to input CSV file
            output_dir (str): Directory to save output files
            
        Returns:
            dict: Dictionary containing the summary DataFrames
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the input CSV
        processed_df = self.process_csv(
            input_csv, 
            output_csv=f"{output_dir}/{os.path.basename(input_csv).replace('.csv', '_with_keywords.csv')}"
        )
        
        # Generate keyword summaries
        summaries = self.generate_keyword_summaries(processed_df, output_dir)
        
        print(f"Total papers: {len(processed_df)}")
        print(f"Papers with all keywords: {len(summaries['all_true'])}")
        print(f"Papers with functional_testing and llm: {len(summaries['functional_testing_llm_true'])}")
        print(f"Papers with functional_testing and functional: {len(summaries['functional_testing_functional_true'])}")
        print(f"Total unique papers in merged results: {len(summaries['merged'])}")
        
        return summaries


# Example usage
if __name__ == "__main__":
    filter_tool = KeywordFilter()
    # Example: Process a CSV file and generate summaries
    type_name = "arxiv"
    # Create a dummy CSV for testing if it doesn't exist
    input_csv_path = f"crawl_results/arxiv/all_arxiv_papers.csv"
    if not os.path.exists(input_csv_path):
        print(f"Creating a dummy input file at: {input_csv_path}")
        os.makedirs(os.path.dirname(input_csv_path), exist_ok=True)
        dummy_data = {
            'title': ['Paper on LLM Test Generation', 'Paper on Web Testing', 'Another Paper'],
            'abstract': [
                'This paper discusses using a large language model (llm) for test case generation. Software testing is critical.',
                'We explore functional testing and GUI testing for web applications. Our new AI agent helps automate the process.',
                'This research focuses on neural networks for image classification.'
            ],
            'submitted': ['2023-10-26', '2023-10-25', '2023-10-24']
        }
        pd.DataFrame(dummy_data).to_csv(input_csv_path, index=False)

    output_dir = f"keywords_summary/{type_name}"
    
    summaries = filter_tool.run_pipeline(input_csv_path, output_dir)