#!/usr/bin/env python3
"""
Paper Downloader Tool

This tool downloads research papers from a consolidated CSV file containing papers
from multiple sources (IEEE, ArXiv, etc.). It supports date filtering and handles
various date formats from different academic databases.

Features:
- Date-based filtering with flexible date format parsing
- Automatic filename generation from paper titles
- Progress tracking with download statistics
- Robust error handling for failed downloads
- Support for limiting download quantities

Author: Research Team
Date: 2025
"""

import pandas as pd
from datetime import datetime
import os
import requests
from tqdm import tqdm
import re

class PaperDownloader:
    def __init__(self, output_dir):
        """
        Initialize the PaperDownloader class.
        
        Args:
            output_dir (str): Directory where downloaded papers will be saved
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created directory: {output_dir}")
    
    def _clean_date_format(self, date_str):
        """
        Clean and standardize date format to help with parsing.
        Handles various date formats from different academic sources:
        - ArXiv: "v1 submitted 15 Mar, 2023" or "originally announced March 2023"
        - IEEE: "22-26 October 2023" or "2023"
        - Other formats: "March 2023", "2023-03-15", etc.
        
        Args:
            date_str: Date string to clean
            
        Returns:
            str: Cleaned date string in YYYY-MM-DD format or None if unparseable
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
        # Handle "originally announced Month Year" format (ArXiv)
        announced_pattern = r'originally announced (\w+)\s+\d{4}'
        announced_match = re.search(announced_pattern, date_str.lower())
        if announced_match:
            month_name = announced_match.group(1).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-01"
        
        # Handle "v1 submitted DD Month, Year" format (ArXiv)
        submitted_pattern = r'submitted\s+(\d{1,2})\s+(\w+)[\s,]+\d{4}'
        submitted_match = re.search(submitted_pattern, date_str.lower())
        if submitted_match:
            day = submitted_match.group(1).zfill(2)  # Pad single digit day with leading zero
            month_name = submitted_match.group(2).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-{day}"
        
        # Handle "DD-DD Month Year" format (IEEE conference dates)
        ieee_date_pattern = r'(\d{1,2})-\d{1,2}\s+(\w+)\s+\d{4}'
        ieee_match = re.search(ieee_date_pattern, date_str.lower())
        if ieee_match:
            day = ieee_match.group(1).zfill(2)
            month_name = ieee_match.group(2).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-{day}"
        
        # Handle "Month Year" format
        month_year_pattern = r'(\w+)\s+\d{4}'
        month_year_match = re.search(month_year_pattern, date_str.lower())
        if month_year_match:
            month_name = month_year_match.group(1).lower()
            if month_name in month_map:
                return f"{year}-{month_map[month_name]}-01"
        
        # Handle already formatted dates like "2023-03-15"
        iso_pattern = r'\d{4}-\d{2}-\d{2}'
        if re.match(iso_pattern, date_str):
            return date_str
        
        # Default: return year with first day
        return f"{year}-01-01"
    
    def filter_papers_by_date(self, input_csv, output_csv=None, start_date="2024-01-01", end_date=None):
        """
        Filter papers by date range.
        
        Args:
            input_csv (str): Path to input CSV file (consolidated papers from multiple sources)
            output_csv (str, optional): Path to save filtered CSV file
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format, defaults to current date
            
        Returns:
            pd.DataFrame: DataFrame containing filtered papers
        """
        try:
            # Load the CSV file
            print(f"📖 Loading papers from: {input_csv}")
            papers = pd.read_csv(input_csv)
            print(f"📊 Loaded {len(papers)} papers from consolidated dataset")
            
            # Check if 'submitted' column exists
            if 'submitted' not in papers.columns:
                print(f"⚠️ Warning: 'submitted' column not found in CSV. Available columns: {', '.join(papers.columns)}")
                print("📝 Using all papers without date filtering.")
                return papers
            
            # Make a copy of the DataFrame to avoid warnings
            papers_copy = papers.copy()
            
            # Try to clean date formats first
            print("🔧 Pre-processing dates for better parsing...")
            papers_copy['cleaned_date'] = papers_copy['submitted'].apply(self._clean_date_format)
            
            # Count how many dates we could clean
            valid_cleaned = papers_copy['cleaned_date'].notna().sum()
            print(f"✅ Successfully cleaned {valid_cleaned} dates out of {len(papers_copy)} entries")
            
            # Convert cleaned dates to datetime objects
            papers_copy['date_obj'] = pd.to_datetime(papers_copy['cleaned_date'], errors='coerce')
            
            # Count valid dates after conversion
            valid_dates_count = papers_copy['date_obj'].notna().sum()
            print(f"📅 Found {valid_dates_count} papers with valid dates out of {len(papers_copy)} total papers")
            
            if valid_dates_count == 0:
                print("⚠️ No valid dates found. Using all papers.")
                return papers
            
            # Define date range
            try:
                start_date_obj = pd.to_datetime(start_date)
                if end_date is None:
                    end_date_obj = datetime.now()
                else:
                    end_date_obj = pd.to_datetime(end_date)
                
                # Filter papers by date range
                filtered_papers = papers_copy[(papers_copy['date_obj'] >= start_date_obj) & 
                                            (papers_copy['date_obj'] <= end_date_obj)]
                
                print(f"🔍 Filtered to {len(filtered_papers)} papers published between {start_date} and {end_date or 'now'}")
                
                if filtered_papers.empty:
                    print("⚠️ No papers found in the specified date range. Using all papers with valid dates.")
                    filtered_papers = papers_copy[papers_copy['date_obj'].notna()]
                    print(f"📋 Using {len(filtered_papers)} papers with valid dates")
                    
                    if filtered_papers.empty:
                        print("⚠️ No papers with valid dates. Using all papers.")
                        filtered_papers = papers
                
                # Remove temporary columns before returning
                if 'cleaned_date' in filtered_papers.columns:
                    filtered_papers = filtered_papers.drop(columns=['cleaned_date'])
                if 'date_obj' in filtered_papers.columns:
                    filtered_papers = filtered_papers.drop(columns=['date_obj'])
                
            except Exception as e:
                print(f"❌ Error parsing dates: {e}")
                print("📝 Using all papers with valid dates")
                filtered_papers = papers
            
            # Save filtered papers to CSV if output_csv is provided
            if output_csv and not filtered_papers.empty:
                filtered_papers.to_csv(output_csv, index=False)
                print(f"💾 Filtered papers saved to {output_csv}")
            
            return filtered_papers
        except Exception as e:
            print(f"❌ Error in filter_papers_by_date: {e}")
            # Return original DataFrame if there's an error
            try:
                return pd.read_csv(input_csv)
            except:
                return pd.DataFrame()
    
    def download_papers(self, papers_df, pdf_link_column='pdf_link', max_papers=None):
        """
        Download papers from URLs in the DataFrame.
        
        Args:
            papers_df (pd.DataFrame): DataFrame containing paper information
            pdf_link_column (str): Column name containing PDF URLs
            max_papers (int, optional): Maximum number of papers to download
            
        Returns:
            list: List of tuples containing (status, file_path) for each download
        """
        results = []
        
        # Check if DataFrame is empty or PDF column doesn't exist
        if papers_df.empty:
            print("📭 No papers to download: empty DataFrame")
            return results
        
        if pdf_link_column not in papers_df.columns:
            print(f"❌ Error: '{pdf_link_column}' column not found in DataFrame")
            available_columns = ", ".join(papers_df.columns)
            print(f"📋 Available columns: {available_columns}")
            return results
        
        # Filter out rows with missing PDF links
        valid_papers = papers_df[papers_df[pdf_link_column].notna()]
        
        # Also filter out empty strings
        valid_papers = valid_papers[valid_papers[pdf_link_column].str.strip() != '']
        
        if valid_papers.empty:
            print(f"📭 No valid PDF links found in '{pdf_link_column}' column")
            return results
        
        print(f"🔗 Found {len(valid_papers)} papers with valid PDF links")
        
        # Limit number of downloads if specified
        if max_papers and max_papers < len(valid_papers):
            valid_papers = valid_papers.head(max_papers)
            print(f"🎯 Limiting downloads to {max_papers} papers")
        
        # Download papers with progress bar
        print(f"📥 Starting download of {len(valid_papers)} papers...")
        for index, row in tqdm(valid_papers.iterrows(), total=len(valid_papers), desc="Downloading papers"):
            pdf_url = row[pdf_link_column]
            try:
                # Generate filename from title if available, otherwise use URL
                if 'title' in row and pd.notna(row['title']) and str(row['title']).strip():
                    # Clean title to use as filename
                    title = str(row['title']).strip()
                    # Remove special characters and limit length
                    title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
                    title = title[:100]  # Limit length
                    filename = f"{title}.pdf"
                else:
                    # Extract filename from URL or use index
                    url_parts = pdf_url.split('/')
                    filename = url_parts[-1] if url_parts[-1].endswith('.pdf') else f"paper_{index}.pdf"
                
                # Ensure unique filenames
                file_path = os.path.join(self.output_dir, filename)
                counter = 1
                original_path = file_path
                while os.path.exists(file_path):
                    name, ext = os.path.splitext(original_path)
                    file_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                # Download file
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                results.append((True, file_path))
                
            except Exception as e:
                print(f"\n❌ Error downloading {pdf_url}: {e}")
                results.append((False, pdf_url))
        
        successful_downloads = sum(1 for status, _ in results if status)
        print(f"\n✅ Successfully downloaded {successful_downloads} out of {len(valid_papers)} papers")
        
        return results

    def download_2023_papers(self, input_csv=None, output_subdir="2023_papers"):
        """
        Filter all papers from 2023 in the consolidated CSV and download their PDFs to a single directory.
        
        Args:
            input_csv (str, optional): Path to the consolidated CSV file. 
                                     Defaults to 'filtered_papers/all_papers_consolidated_unique.csv'
            output_subdir (str): Subdirectory name for saving 2023 papers
            
        Returns:
            dict: Summary of the download process including:
                  - total_found: Number of 2023 papers found
                  - with_pdfs: Number of papers with PDF links
                  - successfully_downloaded: Number of successfully downloaded PDFs
                  - failed_downloads: Number of failed downloads
                  - output_directory: Path where PDFs were saved
        """
        print("🗓️ Starting 2023 Papers Download Process")
        print("=" * 50)
        
        # Use default CSV file if not provided
        if input_csv is None:
            input_csv = "filtered_papers/all_papers_consolidated_unique.csv"
        
        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"❌ Error: Input CSV file not found at: {input_csv}")
            print("💡 Please ensure the consolidated CSV file exists.")
            return {
                'total_found': 0,
                'with_pdfs': 0,
                'successfully_downloaded': 0,
                'failed_downloads': 0,
                'output_directory': None,
                'error': 'Input file not found'
            }
        
        # Create output directory for 2023 papers
        output_2023_dir = os.path.join(self.output_dir, output_subdir)
        if not os.path.exists(output_2023_dir):
            os.makedirs(output_2023_dir)
            print(f"📁 Created directory for 2023 papers: {output_2023_dir}")
        
        # Temporarily change output directory to the 2023 subdirectory
        original_output_dir = self.output_dir
        self.output_dir = output_2023_dir
        
        try:
            # Filter papers from 2023
            print("🔍 Filtering papers from 2023...")
            papers_2023 = self.filter_papers_by_date(
                input_csv=input_csv,
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            total_found = len(papers_2023)
            print(f"📊 Found {total_found} papers from 2023")
            
            if total_found == 0:
                print("📭 No papers from 2023 found in the dataset")
                return {
                    'total_found': 0,
                    'with_pdfs': 0,
                    'successfully_downloaded': 0,
                    'failed_downloads': 0,
                    'output_directory': output_2023_dir,
                    'message': 'No papers from 2023 found'
                }
            
            # Check how many have PDF links
            papers_with_pdfs = papers_2023[papers_2023['pdf_link'].notna()]
            papers_with_pdfs = papers_with_pdfs[papers_with_pdfs['pdf_link'].str.strip() != '']
            with_pdfs = len(papers_with_pdfs)
            print(f"🔗 {with_pdfs} papers have valid PDF links")
            
            if with_pdfs == 0:
                print("📭 No papers from 2023 have valid PDF links")
                return {
                    'total_found': total_found,
                    'with_pdfs': 0,
                    'successfully_downloaded': 0,
                    'failed_downloads': 0,
                    'output_directory': output_2023_dir,
                    'message': 'No PDF links available for 2023 papers'
                }
            
            # Download all PDFs from 2023
            print(f"📥 Downloading PDFs for all {with_pdfs} papers from 2023...")
            download_results = self.download_papers(papers_2023)
            
            # Count successful and failed downloads
            successful_downloads = sum(1 for status, _ in download_results if status)
            failed_downloads = len(download_results) - successful_downloads
            
            # Save list of 2023 papers to CSV
            papers_list_csv = os.path.join(output_2023_dir, "papers_2023_list.csv")
            papers_2023.to_csv(papers_list_csv, index=False)
            print(f"📋 Saved list of 2023 papers to: {papers_list_csv}")
            
            # Create summary report
            summary = {
                'total_found': total_found,
                'with_pdfs': with_pdfs,
                'successfully_downloaded': successful_downloads,
                'failed_downloads': failed_downloads,
                'output_directory': output_2023_dir,
                'papers_list_csv': papers_list_csv
            }
            
            print("=" * 50)
            print("📊 2023 Papers Download Summary:")
            print(f"   📚 Total papers from 2023: {total_found}")
            print(f"   🔗 Papers with PDF links: {with_pdfs}")
            print(f"   ✅ Successfully downloaded: {successful_downloads}")
            print(f"   ❌ Failed downloads: {failed_downloads}")
            print(f"   📁 Output directory: {output_2023_dir}")
            print("=" * 50)
            
            if successful_downloads > 0:
                print("🎉 Download completed! All 2023 papers with available PDFs have been downloaded.")
            else:
                print("⚠️ No PDFs were successfully downloaded.")
            
            return summary
            
        except Exception as e:
            print(f"❌ Error during 2023 papers download: {e}")
            return {
                'total_found': 0,
                'with_pdfs': 0,
                'successfully_downloaded': 0,
                'failed_downloads': 0,
                'output_directory': output_2023_dir,
                'error': str(e)
            }
        
        finally:
            # Restore original output directory
            self.output_dir = original_output_dir
    
    def run_pipeline(self, input_csv, output_dir=None, start_date="2023-01-01", end_date=None, max_papers=None):
        """
        Run the complete pipeline: filter papers and download them.
        
        Args:
            input_csv (str): Path to input CSV file (consolidated papers from multiple sources)
            output_dir (str, optional): Directory to save downloaded papers
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            max_papers (int, optional): Maximum number of papers to download
            
        Returns:
            tuple: (filtered_papers DataFrame, download_results list)
        """
        print("🚀 Starting Paper Download Pipeline")
        print("=" * 50)
        
        if output_dir:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Created output directory: {output_dir}")
        
        # Filter papers by date
        print(f"📅 Filtering papers from {start_date} to {end_date or 'now'}")
        filtered_papers = self.filter_papers_by_date(input_csv, start_date=start_date, end_date=end_date)
        
        # Download papers
        print(f"📥 Starting download process...")
        download_results = self.download_papers(filtered_papers, max_papers=max_papers)
        
        print("=" * 50)
        print("✅ Pipeline completed successfully!")
        
        return filtered_papers, download_results


# Convenience function for easy access
def download_all_2023_papers(output_dir="filtered_papers/downloaded_papers", 
                            input_csv=None):
    """
    Convenience function to download all papers from 2023.
    
    Args:
        output_dir (str): Directory where papers will be downloaded
        input_csv (str, optional): Path to consolidated CSV file
        
    Returns:
        dict: Summary of download process
    """
    downloader = PaperDownloader(output_dir=output_dir)
    return downloader.download_2023_papers(input_csv=input_csv)


# Example usage and testing
if __name__ == "__main__":
    # Initialize downloader with output directory
    downloader = PaperDownloader(output_dir="filtered_papers/downloaded_papers")
    
    # Use the new consolidated file path
    consolidated_file = "filtered_papers/all_papers_consolidated_unique.csv"
    
    # Check if the consolidated file exists
    if os.path.exists(consolidated_file):
        print("🧪 Running example: Download all 2023 papers...")
        
        # Example 1: Download all papers from 2023
        summary = downloader.download_2023_papers()
        
        print("\n🧪 Running example: Download recent papers (2024+)...")
        
        # Example 2: Filter and download recent papers from the consolidated CSV file
        filtered_papers, results = downloader.run_pipeline(
            input_csv=consolidated_file,
            start_date="2024-01-01",  # Recent papers
            max_papers=5  # Limit for testing
        )
        
        successful_downloads = sum(1 for status, _ in results if status)
        print(f"📊 Final Results: {successful_downloads} papers downloaded successfully")
        
    else:
        print(f"⚠️ Consolidated file not found at: {consolidated_file}")
        print("💡 Please run the paper combination tool first to create the consolidated CSV file.")
        print("📖 Example paths to try:")
        print("   - filtered_papers/all_papers_consolidated_unique.csv")
        print("   - output/arxiv/all_arxiv_papers.csv")
        print("   - output/ieee/all_ieee_papers_detailed.csv")
        print("\n📝 You can still test the 2023 download method with:")
        print("   downloader = PaperDownloader('test_output')")
        print("   summary = downloader.download_2023_papers('path/to/your/csv')") 