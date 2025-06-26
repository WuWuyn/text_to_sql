import pandas as pd
from datetime import datetime
from DrissionPage import SessionPage
import os

class PaperDownloader:
    def __init__(self, output_dir="arxiv"):
        """
        Initialize the PaperDownloader class.
        
        Args:
            output_dir (str): Directory where downloaded papers will be saved
        """
        self.output_dir = output_dir
        self.session_page = SessionPage()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    def filter_papers_by_date(self, input_csv, output_csv=None, start_date="2023-01-01", end_date=None):
        """
        Filter papers by date range.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str, optional): Path to save filtered CSV file
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format, defaults to current date
            
        Returns:
            pd.DataFrame: DataFrame containing filtered papers
        """
        # Load the CSV file
        papers = pd.read_csv(input_csv)
        
        # Convert submitted column to datetime
        papers['submitted'] = pd.to_datetime(papers['submitted'], format='%Y-%m-%d')
        
        # Define date range
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Filter papers by date range
        filtered_papers = papers[(papers['submitted'] >= start_date) & (papers['submitted'] <= end_date)]
        
        # Save filtered papers to CSV if output_csv is provided
        if output_csv:
            filtered_papers.to_csv(output_csv, index=False)
            print(f"Filtered papers saved to {output_csv}")
        
        return filtered_papers
    
    def download_papers(self, papers_df, pdf_link_column='pdf_link'):
        """
        Download papers from URLs in the DataFrame.
        
        Args:
            papers_df (pd.DataFrame): DataFrame containing paper information
            pdf_link_column (str): Column name containing PDF URLs
            
        Returns:
            list: List of tuples containing (status, file_path) for each download
        """
        results = []
        
        for pdf_url in papers_df[pdf_link_column]:
            if pd.notna(pdf_url):
                print(f"Downloading: {pdf_url}")
                result = self.session_page.download(pdf_url, self.output_dir)
                print(result)
                results.append(result)
        
        return results
    
    def run_pipeline(self, input_csv, output_dir=None, start_date="2023-01-01", end_date=None):
        """
        Run the complete pipeline: filter papers and download them.
        
        Args:
            input_csv (str): Path to input CSV file
            output_dir (str, optional): Directory to save downloaded papers
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            tuple: (filtered_papers DataFrame, download_results list)
        """
        if output_dir:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # Filter papers by date
        filtered_papers = self.filter_papers_by_date(input_csv, start_date=start_date, end_date=end_date)
        
        # Download papers
        download_results = self.download_papers(filtered_papers)
        
        return filtered_papers, download_results


# Example usage
if __name__ == "__main__":
    downloader = PaperDownloader(output_dir="arxiv")
    
    # Example: Filter papers from a CSV file and download them
    filtered_papers, results = downloader.run_pipeline(
        input_csv="../raw_crawl_papers/acm/all_acm_papers.csv",
        start_date="2023-01-01"
    )
    
    print(f"Downloaded {len(results)} papers") 