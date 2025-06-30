"""
MDPI Paper Crawler

This module provides a class-based approach to crawl MDPI for academic papers
based on specified keywords and extract detailed information from each paper.
Supports multithreading for faster crawling.
"""

from DrissionPage import ChromiumPage, ChromiumOptions
import time
import pandas as pd
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re
import random


class MDPICrawler:
    """
    A class for crawling MDPI papers based on specified keywords.
    
    This class provides methods to search for papers, extract detailed information,
    and save results to CSV files.
    """
    
    def __init__(self, headless=False, output_dir='mdpi', max_threads=4, keyword_sets=None):
        """
        Initialize the MDPI Crawler.
        
        Args:
            headless (bool): Whether to run browser in headless mode
            output_dir (str): Directory to save output CSV files
            max_threads (int): Maximum number of threads for concurrent crawling
            keyword_sets (dict): Dictionary containing keyword sets, e.g.:
                {
                    't2sql': ['keyword1', 'keyword2'],
                    'security': ['security1', 'security2'],
                    'llm': ['llm1', 'llm2'],
                    'custom_category': ['custom1', 'custom2']
                }
        """
        self.headless = headless
        self.output_dir = output_dir
        self.max_threads = max_threads
        self.browser = None
        self.page = None
        
        # Thread management
        self.file_lock = Lock()  # For thread-safe file operations
        self.browser_semaphore = threading.Semaphore(max_threads)  # Limit concurrent browsers
        
        # Keyword sets - store as dictionary for flexibility
        self.keyword_sets = keyword_sets or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __enter__(self):
        """Context manager entry."""
        # Only initialize main browser for single-threaded mode
        # For multithreading, browsers will be created per thread
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _init_browser(self):
        """Initialize the main browser instance (for single-threaded mode only)."""
        if self.browser is None:
            option = ChromiumOptions().auto_port()
            if self.headless:
                option.headless(on_off=True)
            self.browser = ChromiumPage(option)
            self.page = self.browser
            print(f"🔵 Created main Chrome instance on port {self.browser.address[1]}")
    
    def close(self):
        """Close main browser if exists."""
        if self.browser:
            self.browser.quit()
            self.browser = None
            self.page = None
    
    def _create_browser(self):
        """Create a new browser instance with unique port."""
        option = ChromiumOptions().auto_port()
        if self.headless:
            option.headless(on_off=True)
        browser = ChromiumPage(option)
        print(f"🔵 Created Chrome instance on port {browser.address[1]} for thread {threading.current_thread().name}")
        return browser
    
    def _close_browser(self, browser):
        """Close and clean up a browser instance."""
        try:
            port = browser.address[1]
            browser.quit()
            print(f"🔴 Closed Chrome instance on port {port} for thread {threading.current_thread().name}")
        except Exception as e:
            print(f"Error closing browser: {e}")
    
    @staticmethod
    def combination_keywords(sets):
        """Tạo danh sách các chuỗi từ khóa từ một danh sách các bộ từ khóa."""
        if not sets:
            return []
        combinations = itertools.product(*sets)
        return [' AND '.join(combo) for combo in combinations]
    
    def generate_all_combinations(self, keyword_sets=None, primary_key='llm'):
        """
        Tạo danh sách tất cả các tổ hợp từ khóa theo các trường hợp yêu cầu.
        
        Args:
            keyword_sets (dict): Dictionary of keyword sets (uses instance keyword_sets if None)
            primary_key (str): Primary keyword category that must be included in all combinations
            
        Returns:
            list: List of all keyword combinations
        """
        # Use provided keyword sets or fallback to instance keyword sets
        keyword_sets = keyword_sets or self.keyword_sets
        
        # Validate that primary keywords are available
        if not keyword_sets or primary_key not in keyword_sets:
            raise ValueError(f"'{primary_key}' keywords must be provided in keyword_sets")
        
        primary_keywords = keyword_sets[primary_key]
        if not primary_keywords:
            raise ValueError(f"'{primary_key}' keyword list cannot be empty")
        
        # Get all other keyword categories (excluding primary)
        other_categories = {k: v for k, v in keyword_sets.items() 
                          if k != primary_key and v}  # Only non-empty lists
        
        # Validate there's at least one other category
        if not other_categories:
            raise ValueError(f"At least one category besides '{primary_key}' is needed for combinations")
        
        # Generate all possible combinations
        cases = []
        
        # No longer include primary keywords only
        # cases.append([primary_keywords])
        
        # Generate combinations with other categories
        # Use itertools to generate all possible combinations of additional categories
        from itertools import combinations
        
        category_names = list(other_categories.keys())
        
        # For each possible size of combination (1 to all categories)
        for r in range(1, len(category_names) + 1):
            for category_combo in combinations(category_names, r):
                # Create case with primary + selected categories
                case = [primary_keywords]
                for category in category_combo:
                    case.append(other_categories[category])
                cases.append(case)
        
        # Tạo và hợp nhất tất cả các tổ hợp
        all_combinations = []
        for case in cases:
            all_combinations.extend(self.combination_keywords(case))
        
        return all_combinations
    
    def extract_article_info(self, keyword: str, browser=None):
        """
        Extract article information from MDPI for a given keyword.
        
        Args:
            keyword (str): Search keyword
            browser: Browser instance to use (if None, uses default)
            
        Returns:
            list: List of article information dictionaries
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
        
        # Format keyword for URL
        keyword_formatted = keyword.strip().replace(" ", "+")
        url = f"https://www.mdpi.com/search?q={keyword_formatted}"
        
        try:
            page.get(url)
            time.sleep(random.randint(2, 3))
            
            articles_info = []
            
            while True:
                # Tìm tất cả các cụm bài báo
                article_blocks = page.eles("css:div.generic-item.article-item")
                
                if not article_blocks:
                    print(f"No articles found for keyword: {keyword}")
                    break
                
                for block in article_blocks:
                    # 1. Trích xuất liên kết bài báo và tiêu đề bài báo
                    link_element = block.ele("css:a.title-link")
                    link = link_element.attr('href') if link_element else None
                    title = link_element.text.strip() if link_element else None
                    if link and not link.startswith('http'):
                        link = "https://www.mdpi.com" + link
                    
                    # 2. Trích xuất liên kết PDF
                    pdf_link_element = block.ele("css:a.UD_Listings_ArticlePDF")
                    pdf_link = pdf_link_element.attr('href') if pdf_link_element else None
                    if pdf_link and not pdf_link.startswith('http'):
                        pdf_link = "https://www.mdpi.com" + pdf_link
                    
                    # 3. Trích xuất tác giả
                    authors_element = block.ele("css:div.authors")
                    authors = authors_element.text if authors_element else None
                    if authors:
                        authors = authors.replace("by", "").strip()
                    
                    # 4. Trích xuất tóm tắt đầy đủ
                    abstract_full_element = block.ele("css:div.abstract-full")
                    abstract = abstract_full_element.text if abstract_full_element else None
                    if abstract:
                        # Loại bỏ phần "Full article" ở cuối
                        abstract = re.sub(r'\s*Full article$', '', abstract).strip()
                    
                    # 5. Trích xuất ngày gửi bài
                    submitted_element = block.ele("css:div.color-grey-dark")
                    submitted_text = submitted_element.text if submitted_element else None
                    submitted_date = None
                    if submitted_text:
                        # Tìm ngày có định dạng như "19 Mar 2025"
                        match = re.search(r'\d{1,2} \w{3} \d{4}', submitted_text)
                        if match:
                            submitted_date = match.group(0)
                    
                    # 6. Trích xuất DOI
                    doi_element = block.ele("css:a[href^='https://doi.org']")
                    doi = doi_element.attr('href') if doi_element else None
                    
                    # Lưu thông tin vào dictionary
                    article_info = {
                        "link": link,
                        "pdf_link": pdf_link,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "submitted_date": submitted_date,
                        "doi": doi
                    }
                    articles_info.append(article_info)
                    
                    print(f'[Thread {threading.current_thread().name}] Found article: {title}')
                
                # Tìm liên kết chuyển trang
                next_page_link = page.ele("css:a[href*='page_no'] i.material-icons:contains('chevron_right')")
                if next_page_link:
                    # Nhấp vào liên kết chuyển trang
                    next_page_link.click()
                    time.sleep(random.randint(2, 3))
                else:
                    # Không còn liên kết chuyển trang, thoát vòng lặp
                    break
                    
        except Exception as e:
            print(f"Error extracting articles for keyword {keyword}: {str(e)}")
            return []
        
        return articles_info
    
    def _crawl_keyword_worker(self, keyword):
        """Worker function for multithreaded keyword crawling."""
        # Acquire semaphore to limit concurrent browsers
        self.browser_semaphore.acquire()
        browser = None
        try:
            # Create browser for this thread
            browser = self._create_browser()
            print(f'[Thread {threading.current_thread().name}] Processing keyword: {keyword}')
            
            articles = self.extract_article_info(keyword, browser)
            
            # Thread-safe file saving
            if articles:
                partly = pd.DataFrame(articles)
                partly.drop_duplicates(subset=['link'], inplace=True)
                
                keyword_filename = keyword.replace('"', '').replace(' ', '_')
                output_file = f'{self.output_dir}/crawl_by_{keyword_filename}.csv'
                
                with self.file_lock:
                    partly.to_csv(output_file, index=False)
                    print(f'[Thread {threading.current_thread().name}] Saved {len(articles)} articles for keyword: {keyword}')
            
            return keyword, articles
            
        except Exception as e:
            print(f"Error in thread for keyword {keyword}: {str(e)}")
            return keyword, []
        finally:
            # Always close browser and release semaphore
            if browser:
                self._close_browser(browser)
            self.browser_semaphore.release()
    
    def crawl_papers(self, keywords=None, use_multithreading=True):
        """
        Main method to crawl MDPI for papers based on keywords.
        
        Args:
            keywords (list): List of keywords to search (uses default combinations if None)
            use_multithreading (bool): Whether to use multithreading for crawling
            
        Returns:
            str: Path to the final CSV file containing all papers
        """
        if keywords is None:
            keywords = self.generate_all_combinations()
        
        # Initialize empty list to store all results
        all_articles = []
        
        if use_multithreading and self.max_threads > 1:
            print(f"Starting multithreaded crawling with max {self.max_threads} concurrent browsers...")
            
            # Use ThreadPoolExecutor for multithreading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all keyword crawling tasks
                future_to_keyword = {
                    executor.submit(self._crawl_keyword_worker, keyword): keyword 
                    for keyword in keywords
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        _, articles = future.result()
                        all_articles.extend(articles)
                        print(f"Completed crawling for keyword: {keyword} ({len(articles)} articles)")
                    except Exception as e:
                        print(f"Error crawling keyword {keyword}: {str(e)}")
        else:
            # Single-threaded approach
            print("Starting single-threaded crawling...")
            if not self.page:
                self._init_browser()
            
            for keyword in keywords:
                print(f'{keyword} is processed......')
                
                articles = self.extract_article_info(keyword)
                
                # Create DataFrame after collecting all data
                if articles:
                    partly = pd.DataFrame(articles)
                    partly.drop_duplicates(subset=['link'], inplace=True)
                    keyword_filename = keyword.replace('"', '').replace(' ', '_')
                    partly.to_csv(f'{self.output_dir}/crawl_by_{keyword_filename}.csv', index=False)
                
                all_articles.extend(articles)
        
        # Create final DataFrame with all unique articles
        if all_articles:
            full = pd.DataFrame(all_articles)
            full.drop_duplicates(subset=['link'], inplace=True)
            
            output_file = f'{self.output_dir}/all_mdpi_papers.csv'
            full.to_csv(output_file, index=False)
            
            print(f"Total papers collected: {len(full)}")
            print(f"Results saved to: {output_file}")
            
            return output_file
        else:
            print("No papers found!")
            return None
    
    def crawl_complete(self, keywords=None, use_multithreading=True):
        """
        Complete crawling workflow: get paper information.
        
        Args:
            keywords (list): List of keywords to search (uses default combinations if None)
            use_multithreading (bool): Whether to use multithreading
            
        Returns:
            pd.DataFrame: DataFrame with paper information
        """
        print("Starting MDPI crawler...")
        
        # Step: Crawl papers
        output_file = self.crawl_papers(keywords, use_multithreading=use_multithreading)
        
        if output_file:
            papers = pd.read_csv(output_file)
            print("MDPI crawling completed successfully!")
            return papers
        else:
            print("MDPI crawling completed with no results!")
            return pd.DataFrame()


def main():
    """Main execution function using the MDPICrawler class."""
    print("🚀 Starting MDPI Crawler with Limited Chrome Instances")
    print("=" * 60)
    
    # Define keyword sets as dictionary - easy to extend with new categories
    keyword_sets = {
        't2sql': [
            '"text-to-sql"', '"nl2sql"', '"t2sql"', '"text2sql"'] 
        #     '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'
        # ],
        # 'security': [
        #     '"security"', '"access control"', '"injection"', '"prompt injection"', 
        #     '"defense"', '"attack"', '"vulnerability"'
        # ],
        # 'llm': ['"llm"', '"large language model"'],
        # Easy to add new categories in the future:
        # 'ai': ['"artificial intelligence"', '"machine learning"'],
        # 'database': ['"database"', '"sql"', '"query"']
    }
    
    # Initialize crawler with keyword sets
    with MDPICrawler(
        headless=False, 
        output_dir='mdpi', 
        max_threads=4,
        keyword_sets=keyword_sets
    ) as crawler:
        return crawler.crawl_complete(use_multithreading=True)


if __name__ == "__main__":
    main()