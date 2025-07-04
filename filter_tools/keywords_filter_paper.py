import pandas as pd
from fuzzywuzzy import fuzz
import re
from joblib import Parallel, delayed
import os
from collections import Counter
import numpy as np
from datetime import datetime
from typing import Tuple,List, Dict, Any, Union, Optional

class KeywordFilter:
    def __init__(self, input_dir=None, output_dir=None):
        """
        Initialize the KeywordFilter class.
        
        Args:
            input_dir (str, optional): Directory containing input CSV files
            output_dir (str, optional): Directory to save output CSV files
        """
        # Define default keyword groups
        self.functional_testing_keywords = [
            "Software Testing", "Testing", "Test Automation", "Test Case Generation",
            "Test Script Generation", "Test Data Generation", "Test Oracle Generation", "Test Repair"
        ]
        self.functional_keywords = [
            "Functional Testing", "System Testing", "End-to-End Testing", "GUI Testing", 
            "UI Testing", "Web Testing", "Mobile Testing", "Agent", "AI Agent", 
            "Autonomous Agent", "Prompt Engineering", "Chain-of-Thought", "Retrieval-Augmented Generation"
        ]
        self.llm_keywords = ["llm", "large language model"]
        
        # Store input and output directories
        self.input_dir = input_dir
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Configure similarity thresholds
        self.similarity_thresholds = {
            "exact": 100,
            "high": 90,
            "medium": 80,
            "low": 70
        }
            
        print("KeywordFilter initialized. Using built-in sentence tokenizer and n-gram generator.")

    def _clean_date_format(self, date_str):
        """
        Clean and standardize date format to help with parsing.
        
        Args:
            date_str: Date string to clean
            
        Returns:
            str: Cleaned date string or None if unparseable
        """
        if pd.isna(date_str) or not date_str:
            return None
            
        # Convert to string if it's not already
        date_str = str(date_str).strip()
        
        # Month name to number mapping
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Extract year
        year_pattern = r'(19|20)\d{2}'
        year_match = re.search(year_pattern, date_str)
        if not year_match:
            return None
        
        year = year_match.group(0)
        
        # Try to extract month and day
        # Handle "originally announced Month Year" format
        announced_pattern = r'originally announced (\w+)\s+\d{4}'
        announced_match = re.search(announced_pattern, date_str.lower())
        if announced_match:
            month_name = announced_match.group(1).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-01"
        
        # Handle "v1 submitted DD Month, Year" format
        submitted_pattern = r'submitted\s+(\d{1,2})\s+(\w+)[\s,]+\d{4}'
        submitted_match = re.search(submitted_pattern, date_str.lower())
        if submitted_match:
            day = submitted_match.group(1).zfill(2)  # Pad single digit day with leading zero
            month_name = submitted_match.group(2).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-{day}"
        
        # Handle "Month Year" format
        month_year_pattern = r'(\w+)\s+\d{4}'
        month_year_match = re.search(month_year_pattern, date_str.lower())
        if month_year_match:
            month_name = month_year_match.group(1).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-01"
        
        # Default: return year with first day
        return f"{year}-01-01"

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Enhanced sentence tokenizer using regex.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of sentences
        """
        if not isinstance(text, str):
            return []
            
        # Handle common abbreviations to avoid incorrect sentence splitting
        text = re.sub(r'([A-Z][a-z]*\.)(?=[A-Z])', r'\1 ', text)  # Handle "U.S.A." type abbreviations
        
        # Split on sentence boundaries (. ! ?) but account for common exceptions
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]

    def set_keywords(self, functional_testing_keywords=None, functional_keywords=None, llm_keywords=None, custom_keywords=None):
        """
        Set custom keywords for each category.
        
        Args:
            functional_testing_keywords (list): List of software testing related keywords
            functional_keywords (list): List of functional testing related keywords
            llm_keywords (list): List of LLM related keywords
            custom_keywords (dict): Dictionary of custom keyword categories
        """
        if functional_testing_keywords:
            self.functional_testing_keywords = functional_testing_keywords
        if functional_keywords:
            self.functional_keywords = functional_keywords
        if llm_keywords:
            self.llm_keywords = llm_keywords
            
        # Support for custom keyword categories
        if custom_keywords and isinstance(custom_keywords, dict):
            for category, keywords in custom_keywords.items():
                setattr(self, f"{category}_keywords", keywords)
    
    def set_similarity_thresholds(self, exact=100, high=90, medium=80, low=70):
        """
        Set custom similarity thresholds.
        
        Args:
            exact (int): Exact match threshold (0-100)
            high (int): High similarity threshold (0-100)
            medium (int): Medium similarity threshold (0-100)
            low (int): Low similarity threshold (0-100)
        """
        self.similarity_thresholds = {
            "exact": exact,
            "high": high,
            "medium": medium,
            "low": low
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()  # Convert to lowercase
        
        # Replace special characters with space but keep hyphens in compound words
        text = re.sub(r'([^\w\s-]|_)', ' ', text)  
        
        # Replace hyphens only when they're not part of compound words
        text = re.sub(r'(\s)-|-(\s)', r'\1 \2', text)
        
        text = re.sub(r'\s+', ' ', text)  # Reduce multiple spaces to one
        text = text.strip()  # Remove leading/trailing spaces
        return text
    
    def generate_ngrams(self, text: str, n: int) -> List[str]:
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
    
    def count_keyword_occurrences(self, sentences: List[str], keywords: List[str], threshold: int = 90) -> Dict[str, int]:
        """
        Count occurrences of each keyword in the text.
        
        Args:
            sentences (list): List of sentences
            keywords (list): List of keywords to check
            threshold (int): Minimum similarity threshold (0-100)
            
        Returns:
            dict: Dictionary of keyword occurrences
        """
        all_ngrams = []
        for sentence in sentences:
            preprocessed_sentence = self.preprocess_text(sentence)
            # Generate n-grams of various lengths
            for n in range(1, 5):  # From 1-gram to 4-gram
                all_ngrams.extend(self.generate_ngrams(preprocessed_sentence, n))
        
        # Preprocess keywords once to avoid redundant work
        processed_keywords = [self.preprocess_text(k) for k in keywords]
        
        # Count occurrences
        keyword_counts = Counter()
        
        for i, keyword in enumerate(processed_keywords):
            original_keyword = keywords[i]
            for ngram in all_ngrams:
                similarity = fuzz.WRatio(keyword, ngram)
                if similarity >= threshold:
                    keyword_counts[original_keyword] += 1
                    
            # Also check against full sentences for broader context matches
            for sentence in sentences:
                similarity = fuzz.WRatio(keyword, self.preprocess_text(sentence))
                if similarity >= threshold:
                    keyword_counts[original_keyword] += 1
                    
        return dict(keyword_counts)
    
    def check_group(self, sentences: List[str], keywords: List[str], threshold: int = 90) -> Tuple[bool, Dict[str, int]]:
        """
        Enhanced function to check if the group of keywords is related to the text.
        
        Args:
            sentences (list): List of sentences
            keywords (list): List of keywords to check
            threshold (int): Minimum similarity threshold (0-100)
            
        Returns:
            tuple: (is_related, keyword_counts) where is_related is a boolean and keyword_counts is a dictionary
        """
        keyword_counts = self.count_keyword_occurrences(sentences, keywords, threshold)
        return any(count > 0 for count in keyword_counts.values()), keyword_counts
    
    def classify_abstract(self, abstract: str, min_keyword_occurrences: int = 1) -> Dict[str, Any]:
        """
        Enhanced abstract classification with detailed metrics.
        
        Args:
            abstract (str): Abstract text to classify
            min_keyword_occurrences (int): Minimum number of keyword occurrences to consider relevant
            
        Returns:
            dict: Dictionary with classification results and metrics
        """
        sentences = self.tokenize_sentences(abstract)
        
        # Use medium threshold for initial classification
        threshold = self.similarity_thresholds["medium"]
        
        # Check each keyword group
        functional_testing_result, functional_testing_counts = self.check_group(sentences, self.functional_testing_keywords, threshold)
        functional_result, functional_counts = self.check_group(sentences, self.functional_keywords, threshold)
        llm_result, llm_counts = self.check_group(sentences, self.llm_keywords, threshold)
        
        # Calculate total occurrences for each category
        functional_testing_total = sum(functional_testing_counts.values())
        functional_total = sum(functional_counts.values())
        llm_total = sum(llm_counts.values())
        
        # Consider a category relevant only if it has at least min_keyword_occurrences matches
        functional_testing_relevant = functional_testing_total >= min_keyword_occurrences
        functional_relevant = functional_total >= min_keyword_occurrences
        llm_relevant = llm_total >= min_keyword_occurrences
        
        return {
            "is_functional_testing": functional_testing_relevant,
            "is_functional": functional_relevant,
            "is_llm": llm_relevant,
            "functional_testing_counts": functional_testing_counts,
            "functional_counts": functional_counts,
            "llm_counts": llm_counts,
            "functional_testing_total": functional_testing_total,
            "functional_total": functional_total,
            "llm_total": llm_total,
            "relevance_score": (functional_testing_total + functional_total + llm_total) / 3
        }
    
    def process_csv(self, input_csv: str, output_csv: Optional[str] = None, min_keyword_occurrences: int = 1) -> pd.DataFrame:
        """
        Enhanced CSV processing with configurable keyword occurrence threshold.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str, optional): Path to save output CSV file
            min_keyword_occurrences (int): Minimum number of keyword occurrences to consider relevant
            
        Returns:
            pd.DataFrame: Updated DataFrame with new keyword columns
        """
        df = pd.read_csv(input_csv)
        
        if 'abstract' not in df.columns:
            raise ValueError("CSV file must contain an 'abstract' column.")
        
        df['abstract'] = df['abstract'].fillna('').astype(str)  # Handle missing abstracts
        
        # Process abstracts in parallel
        classifications = Parallel(n_jobs=-1)(
            delayed(self.classify_abstract)(abstract, min_keyword_occurrences) 
            for abstract in df['abstract']
        )
        
        # Extract results
        df['functional_testing'] = [result["is_functional_testing"] for result in classifications]
        df['functional'] = [result["is_functional"] for result in classifications]
        df['llm'] = [result["is_llm"] for result in classifications]
        
        # Add new metrics columns
        df['functional_testing_count'] = [result["functional_testing_total"] for result in classifications]
        df['functional_count'] = [result["functional_total"] for result in classifications]
        df['llm_count'] = [result["llm_total"] for result in classifications]
        df['relevance_score'] = [result["relevance_score"] for result in classifications]
        
        # Keep old column names for compatibility
        df['t2sql'] = df['functional_testing']  
        df['security'] = df['functional']
        
        # Sort by relevance score and submission date if available
        sort_columns = []
        if 'relevance_score' in df.columns:
            sort_columns.append('relevance_score')
            
        if 'submitted' in df.columns:
            # Clean date formats first
            df['cleaned_date'] = df['submitted'].apply(self._clean_date_format)
            
            # Convert cleaned dates to datetime objects
            df['date_obj'] = pd.to_datetime(df['cleaned_date'], errors='coerce')
            
            # Use date_obj for sorting
            sort_columns.append('date_obj')
            
        if sort_columns:
            df = df.sort_values(by=sort_columns, ascending=[False, False])
            
        # Remove temporary date columns before saving
        if 'cleaned_date' in df.columns:
            df = df.drop(columns=['cleaned_date'])
        if 'date_obj' in df.columns:
            df = df.drop(columns=['date_obj'])

        if output_csv:
            # Ensure directory exists
            output_dir = os.path.dirname(output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_csv, index=False)
        
        return df
    
    def filter_papers(self, keywords: List[str] = None, min_keyword_occurrences: int = 1, input_files: List[str] = None) -> List[str]:
        """
        Filter papers by keywords with configurable minimum occurrences.
        
        Args:
            keywords (list): Custom keywords to filter by (optional)
            min_keyword_occurrences (int): Minimum number of keyword occurrences
            input_files (list): List of input CSV files (optional, if None, all CSV files in input_dir will be used)
            
        Returns:
            list: List of filtered output CSV files
        """
        if not self.input_dir and not input_files:
            raise ValueError("Either input_dir or input_files must be provided")
            
        if not input_files:
            # Get all CSV files in input_dir
            input_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                          if f.lower().endswith('.csv')]
        
        if not input_files:
            raise ValueError(f"No CSV files found in input directory: {self.input_dir}")
            
        output_files = []
        
        # Set custom keywords if provided
        if keywords:
            custom_keywords = {"custom": keywords}
            self.set_keywords(custom_keywords=custom_keywords)
            
        # Process each input file
        for input_file in input_files:
            base_name = os.path.basename(input_file).replace('.csv', '')
            output_file = os.path.join(self.output_dir, f"{base_name}_filtered.csv") if self.output_dir else None
            
            try:
                df = self.process_csv(input_file, output_file, min_keyword_occurrences)
                
                if output_file:
                    output_files.append(output_file)
                    print(f"Filtered {input_file} -> {output_file}")
                    print(f"  Found {df['functional_testing'].sum()} functional testing papers")
                    print(f"  Found {df['functional'].sum()} functional papers")
                    print(f"  Found {df['llm'].sum()} LLM papers")
                    
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                
        return output_files

    def generate_keyword_summaries(self, df: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Enhanced summary generation with more informative metrics.
        
        Args:
            df (pd.DataFrame): DataFrame with functional_testing, functional, and llm columns
            output_dir (str): Directory to save summary CSV files
            
        Returns:
            dict: Dictionary containing the summary DataFrames
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Add combined relevance field (any of the three categories)
        df['any_relevant'] = df[['functional_testing', 'functional', 'llm']].any(axis=1)
        
        # Generate various combinations
        summary_dfs = {}
        
        # 1. Papers with all three keyword types
        all_true_df = df[(df['functional_testing'] == True) & (df['functional'] == True) & (df['llm'] == True)].copy()
        all_true_df['relevance_type'] = 'all_categories'
        summary_dfs['all_true'] = all_true_df
        
        # 2. Papers with functional_testing and llm keywords
        ft_llm_df = df[(df['functional_testing'] == True) & (df['llm'] == True)].copy()
        ft_llm_df['relevance_type'] = 'functional_testing_and_llm'
        summary_dfs['functional_testing_llm_true'] = ft_llm_df
        
        # 3. Papers with functional_testing and functional keywords
        ft_func_df = df[(df['functional_testing'] == True) & (df['functional'] == True)].copy()
        ft_func_df['relevance_type'] = 'functional_testing_and_functional'
        summary_dfs['functional_testing_functional_true'] = ft_func_df
        
        # 4. Papers with functional and llm keywords
        func_llm_df = df[(df['functional'] == True) & (df['llm'] == True)].copy()
        func_llm_df['relevance_type'] = 'functional_and_llm'
        summary_dfs['functional_llm_true'] = func_llm_df
        
        # 5. Papers with any relevant keyword
        any_df = df[df['any_relevant'] == True].copy()
        any_df['relevance_type'] = 'any_relevant'
        summary_dfs['any_relevant'] = any_df
        
        # Save each summary DataFrame to CSV
        for name, sum_df in summary_dfs.items():
            # Sort by relevance score and submission date if available
            if 'relevance_score' in sum_df.columns:
                # Clean and parse dates for sorting
                if 'submitted' in sum_df.columns:
                    sum_df['cleaned_date'] = sum_df['submitted'].apply(self._clean_date_format)
                    sum_df['date_obj'] = pd.to_datetime(sum_df['cleaned_date'], errors='coerce')
                    sum_df = sum_df.sort_values(by=['relevance_score', 'date_obj'], ascending=[False, False])
                    # Remove temporary date columns
                    if 'cleaned_date' in sum_df.columns:
                        sum_df = sum_df.drop(columns=['cleaned_date'])
                    if 'date_obj' in sum_df.columns:
                        sum_df = sum_df.drop(columns=['date_obj'])
                else:
                    sum_df = sum_df.sort_values(by='relevance_score', ascending=False)
                    
            sum_df.to_csv(f"{output_dir}/{name}.csv", index=False)
            
        # 6. Merge all results and remove duplicates
        merged_df = pd.concat(list(summary_dfs.values()), ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=[col for col in merged_df.columns if col != 'relevance_type'])
        merged_df.to_csv(f"{output_dir}/merged_keyword_results.csv", index=False)
        summary_dfs['merged'] = merged_df
        
        return summary_dfs
    
    def run_pipeline(self, input_csv: str, output_dir: str = None, min_keyword_occurrences: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Run the complete enhanced pipeline: process CSV and generate summaries.
        
        Args:
            input_csv (str): Path to input CSV file
            output_dir (str, optional): Directory to save output files
            min_keyword_occurrences (int): Minimum number of keyword occurrences
            
        Returns:
            dict: Dictionary containing the summary DataFrames
        """
        if output_dir:
            self.output_dir = output_dir
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process the input CSV
        processed_df = self.process_csv(
            input_csv, 
            output_csv=f"{output_dir}/{os.path.basename(input_csv).replace('.csv', '_with_keywords.csv')}",
            min_keyword_occurrences=min_keyword_occurrences
        )
        
        # Generate keyword summaries
        summaries = self.generate_keyword_summaries(processed_df, output_dir)
        
        # Generate and save summary statistics
        summary_stats = {
            "total_papers": len(processed_df),
            "papers_with_all_keywords": len(summaries['all_true']),
            "papers_with_functional_testing_and_llm": len(summaries['functional_testing_llm_true']),
            "papers_with_functional_testing_and_functional": len(summaries['functional_testing_functional_true']),
            "papers_with_functional_and_llm": len(summaries['functional_llm_true']),
            "papers_with_any_keywords": len(summaries['any_relevant']),
            "total_unique_papers_in_merged": len(summaries['merged'])
        }
        
        # Save summary statistics to JSON
        import json
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        # Print summary
        print(f"📊 Summary Statistics:")
        print(f"  • Total papers: {summary_stats['total_papers']}")
        print(f"  • Papers with all keywords: {summary_stats['papers_with_all_keywords']}")
        print(f"  • Papers with functional testing and LLM: {summary_stats['papers_with_functional_testing_and_llm']}")
        print(f"  • Papers with functional testing and functional: {summary_stats['papers_with_functional_testing_and_functional']}")
        print(f"  • Papers with functional and LLM: {summary_stats['papers_with_functional_and_llm']}")
        print(f"  • Papers with any keywords: {summary_stats['papers_with_any_keywords']}")
        print(f"  • Total unique papers in merged results: {summary_stats['total_unique_papers_in_merged']}")
        
        return summaries


# Example usage
if __name__ == "__main__":
    # Initialize with input and output directories
    filter_tool = KeywordFilter(
        input_dir="crawl_results/arxiv",
        output_dir="crawl_results/filtered"
    )
    
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
    
    # Run with custom minimum keyword occurrences
    summaries = filter_tool.run_pipeline(input_csv_path, output_dir, min_keyword_occurrences=2)