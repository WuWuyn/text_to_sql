"""
Duplicate Filter

This module provides functionality to check for duplicates in output files
and combine them into a single file.
"""

import os
import pandas as pd
import glob
from typing import List, Dict, Optional, Union


class DuplicateFilter:
    """
    A class for filtering duplicates from crawler output files and combining them into a single file.
    """
    
    def __init__(self, input_dirs: Union[List[str], str] = None, output_dir: str = None):
        """
        Initialize the DuplicateFilter.
        
        Args:
            input_dirs (Union[List[str], str], optional): Directory or list of directories containing input CSV files
            output_dir (str, optional): Directory to save output CSV files
        """
        # Convert single directory to list
        if isinstance(input_dirs, str):
            self.input_dirs = [input_dirs]
        else:
            self.input_dirs = input_dirs or []
            
        self.output_dir = output_dir or "filtered_papers"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        print(f"Initializing filter with {len(self.input_dirs)} source directories...")
    
    def find_csv_files(self, directory: str = None) -> List[str]:
        """
        Find all CSV files in the specified directory.
        
        Args:
            directory (str, optional): Directory to search for CSV files
            
        Returns:
            List[str]: List of CSV file paths
        """
        if directory is None:
            if not self.input_dirs:
                return []
            
            # Find CSV files in all input directories
            all_files = []
            for input_dir in self.input_dirs:
                files = glob.glob(os.path.join(input_dir, "*.csv"))
                all_files.extend(files)
            return all_files
        else:
            # Find CSV files in the specified directory
            return glob.glob(os.path.join(directory, "*.csv"))
    
    def process_and_filter(self, specific_files: List[str] = None) -> pd.DataFrame:
        """
        Process all CSV files, combine them, and remove duplicates.
        
        Args:
            specific_files (List[str], optional): List of specific CSV files to process
            
        Returns:
            pd.DataFrame: Combined DataFrame with duplicates removed
        """
        print("\nProcessing papers and removing duplicates...")
        
        # Find all CSV files if specific files are not provided
        if specific_files is None:
            all_files = []
            for input_dir in self.input_dirs:
                files = self.find_csv_files(input_dir)
                print(f"Found {len(files)} CSV files in {input_dir}")
                all_files.extend(files)
        else:
            all_files = specific_files
            print(f"Processing {len(all_files)} specified CSV files")
        
        # Read and combine all CSV files
        combined_df = pd.DataFrame()
        for file in all_files:
            try:
                df = pd.read_csv(file)
                print(f"Read {len(df)} rows from {os.path.relpath(file, 'output')}")
                
                # Skip empty DataFrames
                if df.empty:
                    continue
                
                # Ensure all DataFrames have the same columns
                if combined_df.empty:
                    combined_df = df
                else:
                    # Get common columns
                    common_cols = list(set(combined_df.columns) & set(df.columns))
                    if common_cols:
                        combined_df = pd.concat([combined_df[common_cols], df[common_cols]], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if combined_df.empty:
            print("No data found in any of the CSV files.")
            return combined_df
        
        # Count rows before deduplication
        total_rows = len(combined_df)
        print(f"\nTotal rows before deduplication: {total_rows}")
        
        # Remove duplicates based on title (case-insensitive)
        if 'title' in combined_df.columns:
            # Normalize titles for comparison (lowercase, strip whitespace, and remove prefixes like "Poster:")
            combined_df['title_normalized'] = combined_df['title'].str.lower().str.strip()
            
            # Remove common prefixes like "Poster:", "Paper:", etc.
            prefixes_to_remove = ['poster:', 'paper:', 'research:', 'abstract:', 'article:']
            for prefix in prefixes_to_remove:
                combined_df['title_normalized'] = combined_df['title_normalized'].str.replace(f'^{prefix}\\s*', '', regex=True)
            
            # Remove rows with blank titles
            blank_titles = combined_df['title_normalized'].isna() | (combined_df['title_normalized'] == '')
            if blank_titles.any():
                blank_count = blank_titles.sum()
                print(f"Removing {blank_count} entries with blank titles")
                combined_df = combined_df[~blank_titles]
            
            # Check for duplicates before removal
            duplicates = combined_df.duplicated(subset=['title_normalized'], keep=False)
            if duplicates.any():
                duplicate_count = duplicates.sum() // 2  # Each duplicate appears twice in the count
                duplicate_titles = combined_df.loc[duplicates, 'title'].unique()
                print(f"Found {duplicate_count} duplicate titles:")
                for i, title in enumerate(duplicate_titles[:5], 1):
                    print(f"  {i}. {title}")
                if len(duplicate_titles) > 5:
                    print(f"  ... and {len(duplicate_titles) - 5} more")
            
            # Remove duplicates based on normalized title
            combined_df = combined_df.drop_duplicates(subset=['title_normalized'], keep='first')
            
            # Remove the temporary normalized title column
            combined_df = combined_df.drop(columns=['title_normalized'])
            
            # Count rows after deduplication
            unique_rows = len(combined_df)
            duplicates_removed = total_rows - unique_rows
            print(f"Removed {duplicates_removed} duplicate rows based on title")
            print(f"Unique papers after deduplication: {unique_rows}")
        
        return combined_df
    
    def save_consolidated_file(self, df: pd.DataFrame, filename: str = "all_papers_consolidated_no_filtering.csv") -> str:
        """
        Save the consolidated DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str, optional): Output filename
            
        Returns:
            str: Path to the saved file
        """
        if df.empty:
            print("No data to save.")
            return None
        
        # Create output path
        output_path = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} papers to {output_path}")
        
        return output_path
    
    def run(self, specific_files: List[str] = None, output_filename: str = "all_papers_consolidated_no_filtering.csv") -> str:
        """
        Run the entire filtering process.
        
        Args:
            specific_files (List[str], optional): List of specific CSV files to process
            output_filename (str, optional): Output filename
            
        Returns:
            str: Path to the saved file
        """
        # Process and filter
        filtered_df = self.process_and_filter(specific_files)
        
        # Save consolidated file
        return self.save_consolidated_file(filtered_df, output_filename)


if __name__ == "__main__":
    # Example usage
    base_output_dir = "output"
    ieee_dir = os.path.join(base_output_dir, "ieee")
    arxiv_dir = os.path.join(base_output_dir, "arxiv")
    
    # Check if directories exist
    source_dirs = []
    if os.path.exists(ieee_dir):
        source_dirs.append(ieee_dir)
    if os.path.exists(arxiv_dir):
        source_dirs.append(arxiv_dir)
    
    # Initialize filter
    filter = DuplicateFilter(input_dirs=source_dirs, output_dir="filtered_papers")
    
    # Process specific files if needed
    ieee_main_file = os.path.join(ieee_dir, "all_ieee_papers_detailed.csv")
    arxiv_main_file = os.path.join(arxiv_dir, "all_arxiv_papers.csv")
    
    specific_files = []
    if os.path.exists(ieee_main_file):
        specific_files.append(ieee_main_file)
    if os.path.exists(arxiv_main_file):
        specific_files.append(arxiv_main_file)
    
    # Run filter on specific files
    if specific_files:
        print(f"Processing main files: {[os.path.basename(f) for f in specific_files]}")
        filter.run(specific_files=specific_files)
    else:
        # Run filter on all files
        filter.run() 