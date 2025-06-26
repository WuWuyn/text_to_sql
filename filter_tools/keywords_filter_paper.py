import pandas as pd
from fuzzywuzzy import fuzz
import re
from nltk import sent_tokenize
from nltk import ngrams
from joblib import Parallel, delayed
import os

class KeywordFilter:
    def __init__(self):
        """Initialize the KeywordFilter class."""
        # Define default keyword groups
        self.text_to_sql_keywords = [
            "text-to-sql", "nl2sql", "t2sql", "text2sql", 
            "natural language to sql", "semantic parsing to sql", "nl to sql"
        ]
        self.security_keywords = [
            "security", "access control", "injection", "prompt injection", 
            "defense", "attack", "vulnerability"
        ]
        self.llm_keywords = ["llm", "large language model"]
        
        # Ensure NLTK data is available
        try:
            sent_tokenize("Test sentence.")
        except LookupError:
            import nltk
            nltk.download('punkt')
    
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
        Generate n-grams from the text.
        
        Args:
            text (str): Text to generate n-grams from
            n (int): Size of n-grams
            
        Returns:
            list: List of n-grams
        """
        words = text.split()
        return [' '.join(gram) for gram in ngrams(words, n)]
    
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
            for n in range(1, 5):  # From 1-gram to 4-gram
                all_ngrams.extend(self.generate_ngrams(sentence, n))
        
        for keyword in keywords:
            for ngram in all_ngrams:
                if fuzz.WRatio(keyword, ngram) > threshold:
                    return True
            for sentence in sentences:
                if fuzz.WRatio(keyword, sentence) > threshold:
                    return True
        return False
    
    def classify_abstract(self, abstract):
        """
        Classify the abstract and return a list of True/False for each keyword group.
        
        Args:
            abstract (str): Abstract text to classify
            
        Returns:
            list: List of boolean values [t2sql_result, security_result, llm_result]
        """
        abstract = self.preprocess_text(abstract)
        sentences = sent_tokenize(abstract)
        
        t2sql_result = self.check_group(sentences, self.text_to_sql_keywords)
        security_result = self.check_group(sentences, self.security_keywords)
        llm_result = self.check_group(sentences, self.llm_keywords)
        
        return [t2sql_result, security_result, llm_result]
    
    def process_csv(self, input_csv, output_csv=None):
        """
        Read a CSV file, classify each abstract with parallel processing,
        and add 3 new boolean columns: t2sql, security, llm.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str, optional): Path to save output CSV file
            
        Returns:
            pd.DataFrame: Updated DataFrame with new boolean columns
        """
        df = pd.read_csv(input_csv)
        
        if 'abstract' not in df.columns:
            raise ValueError("CSV file must contain an 'abstract' column.")
        
        df['abstract'] = df['abstract'].astype(str)
        classifications = Parallel(n_jobs=-1)(delayed(self.classify_abstract)(abstract) for abstract in df['abstract'])
        
        df['t2sql'] = [result[0] for result in classifications]
        df['security'] = [result[1] for result in classifications]
        df['llm'] = [result[2] for result in classifications]
        
        if 'submitted' in df.columns:
            df['submitted'] = pd.to_datetime(df['submitted'])
            df = df.sort_values(by='submitted', ascending=False)

        if output_csv:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
        
        return df
    
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
        
        # 2. Papers with t2sql and llm keywords
        t2sql_llm_true_df = df[(df['t2sql'] == True) & (df['llm'] == True)].copy()
        if 'submitted' in t2sql_llm_true_df.columns:
            t2sql_llm_true_df['submitted'] = pd.to_datetime(t2sql_llm_true_df['submitted'])
            t2sql_llm_true_df = t2sql_llm_true_df.sort_values(by='submitted', ascending=False)
        t2sql_llm_true_df.to_csv(f"{output_dir}/t2sql_llm_true.csv", index=False)
        
        # 3. Papers with t2sql and security keywords
        t2sql_security_true_df = df[(df['t2sql'] == True) & (df['security'] == True)].copy()
        if 'submitted' in t2sql_security_true_df.columns:
            t2sql_security_true_df['submitted'] = pd.to_datetime(t2sql_security_true_df['submitted'])
            t2sql_security_true_df = t2sql_security_true_df.sort_values(by='submitted', ascending=False)
        t2sql_security_true_df.to_csv(f"{output_dir}/t2sql_security_true.csv", index=False)
        
        # 4. Merge all results and remove duplicates
        merged_df = pd.concat([all_true_df, t2sql_llm_true_df, t2sql_security_true_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates()
        merged_df.to_csv(f"{output_dir}/merged_keyword_results.csv", index=False)
        
        return {
            'all_true': all_true_df,
            't2sql_llm_true': t2sql_llm_true_df,
            't2sql_security_true': t2sql_security_true_df,
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
        print(f"Papers with t2sql and llm: {len(summaries['t2sql_llm_true'])}")
        print(f"Papers with t2sql and security: {len(summaries['t2sql_security_true'])}")
        print(f"Total unique papers in merged results: {len(summaries['merged'])}")
        
        return summaries


# Example usage
if __name__ == "__main__":
    filter_tool = KeywordFilter()
    
    # Example: Process a CSV file and generate summaries
    type_name = "arxiv"
    input_csv = f"../raw_crawl_papers/date_processed/{type_name}_papers.csv"
    output_dir = f"../keywords_summary/{type_name}"
    
    summaries = filter_tool.run_pipeline(input_csv, output_dir) 