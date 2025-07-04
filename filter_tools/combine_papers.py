#!/usr/bin/env python3
"""
Paper Combination and Deduplication Tool

This script combines papers from different sources (IEEE, ArXiv, etc.) and removes duplicates
based on paper titles. It's designed to work with the crispy-chicken research paper collection system.

Author: Research Team
Date: 2025
"""

import pandas as pd
import os
import sys
from pathlib import Path

class PaperCombiner:
    """Handles combining and deduplicating research papers from multiple sources."""
    
    def __init__(self, base_dir="."):
        """Initialize the PaperCombiner with base directory."""
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "filtered_papers"
        self.output_dir.mkdir(exist_ok=True)
    
    def combine_papers(self, file_paths, output_filename="combined_unique_papers.csv"):
        """
        Combine papers from multiple CSV files and remove duplicates.
        
        Args:
            file_paths (list): List of paths to CSV files to combine
            output_filename (str): Name of output file
            
        Returns:
            str: Path to the created output file
        """
        print("=" * 60)
        print("PAPER COMBINATION AND DEDUPLICATION TOOL")
        print("=" * 60)
        
        # Check if all input files exist
        missing_files = [f for f in file_paths if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Error: Missing files: {missing_files}")
            return None
        
        combined_dfs = []
        total_original_papers = 0
        
        # Read each file
        for file_path in file_paths:
            print(f"\n📖 Reading '{file_path}'...")
            try:
                df = pd.read_csv(file_path)
                papers_count = len(df)
                total_original_papers += papers_count
                combined_dfs.append(df)
                print(f"   ✅ Found {papers_count} papers")
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
                return None
        
        # Combine all DataFrames
        print(f"\n🔄 Combining papers...")
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        print(f"   📊 Total papers before deduplication: {len(combined_df)}")
        
        # Remove rows with empty titles
        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['title'])
        after_dropna = len(combined_df)
        
        if initial_count > after_dropna:
            print(f"   🧹 Removed {initial_count - after_dropna} papers with missing titles")
        
        # Remove duplicates based on normalized titles
        print(f"\n🔍 Detecting and removing duplicates...")
        before_dedup = len(combined_df)
        
        # Create normalized titles for comparison
        normalized_titles = combined_df['title'].str.lower().str.strip()
        unique_df = combined_df.loc[~normalized_titles.duplicated(keep='first')].copy()
        
        after_dedup = len(unique_df)
        duplicates_removed = before_dedup - after_dedup
        
        print(f"   🗑️  Removed {duplicates_removed} duplicate papers")
        print(f"   ✨ Final unique papers: {after_dedup}")
        
        # Save the result
        output_path = self.output_dir / output_filename
        unique_df.to_csv(output_path, index=False)
        
        # Summary
        print(f"\n" + "=" * 60)
        print("📈 SUMMARY")
        print("=" * 60)
        print(f"Input sources: {len(file_paths)}")
        print(f"Total original papers: {total_original_papers}")
        print(f"Papers after cleaning: {before_dedup}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Final unique papers: {after_dedup}")
        print(f"Deduplication rate: {(duplicates_removed/before_dedup)*100:.1f}%")
        print(f"Output file: {output_path}")
        print("=" * 60)
        
        return str(output_path)

def main():
    """Main function to run the paper combination process."""
    # Initialize combiner
    combiner = PaperCombiner()
    
    # Define input files - you can modify these paths as needed
    input_files = [
        "output/ieee/all_ieee_papers_detailed.csv",
        "output/arxiv/all_arxiv_papers.csv"
    ]
    
    # Combine papers
    output_file = combiner.combine_papers(
        file_paths=input_files,
        output_filename="all_papers_consolidated_unique.csv"
    )
    
    if output_file:
        print(f"\n✅ SUCCESS: Combined papers saved to {output_file}")
    else:
        print(f"\n❌ FAILED: Could not combine papers")
        sys.exit(1)

if __name__ == "__main__":
    main() 