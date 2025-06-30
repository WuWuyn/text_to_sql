"""
IEEE Xplore Paper Crawler

This module provides a class-based approach to crawl IEEE Xplore for academic papers
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


class IEEECrawler:
    """
    A class for crawling IEEE Xplore papers based on specified keywords.
    
    This class provides methods to search for papers, extract detailed information,
    and save results to CSV files.
    """
    
    def __init__(self, headless=False, output_dir='ieee', max_threads=4, keyword_sets=None):
        """
        Initialize the IEEE Crawler.
        
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
    
    def crawl_title_link(self, keyword: str, browser=None):
        """
        Crawl titles and links from IEEE Xplore for a given keyword.
        
        Args:
            keyword (str): Search keyword
            browser: Browser instance to use (if None, uses default)
            
        Returns:
            list: List of paper links
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
            
        try:
            # Open the IEEE Xplore website
            search_input = f'https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={keyword}'
            page.get(search_input)

            # Wait for the page to load
            time.sleep(2)

            titles = []
            links = []
            cnt = 1

            while True:
                try:
                    # Wait for the page to load
                    time.sleep(5)

                    # Find all article links
                    tags = page.eles('css:a.fw-bold') or page.ele('css:a')
                    
                    if not tags:
                        print(f"No results found on page {cnt} for keyword: {keyword}")
                        break

                    for a_tag in tags:
                        href = a_tag.attr('href')
                        title = a_tag.text
                        if href and title:
                            titles.append(title)
                            links.append(href)
                            print(f'[{keyword}] Page {cnt} - Title: {title}')

                    # Check for next button
                    next_button = page.ele('css:.next-btn')
                    if not next_button:
                        print(f"Reached last page for keyword: {keyword}")
                        break
                    
                    next_button.click()
                    cnt += 1
                    
                except Exception as e:
                    print(f"Error processing page {cnt} for keyword {keyword}: {str(e)}")
                    break

        except Exception as e:
            print(f"Error searching for keyword {keyword}: {str(e)}")
            return []

        return links
    
    def _crawl_keyword_worker(self, keyword):
        """Worker function for multithreaded keyword crawling."""
        # Acquire semaphore to limit concurrent browsers
        self.browser_semaphore.acquire()
        browser = None
        try:
            # Create browser for this thread
            browser = self._create_browser()
            print(f'[Thread {threading.current_thread().name}] Processing keyword: {keyword}')
            
            links = self.crawl_title_link(keyword, browser)
            
            # Thread-safe file saving
            if links:
                partly = pd.DataFrame()
                partly['link'] = links
                partly.drop_duplicates(subset=['link'], inplace=True)
                
                keyword_filename = keyword.replace('"', '').replace(' ', '_')
                output_file = f'{self.output_dir}/crawl_by_{keyword_filename}.csv'
                
                with self.file_lock:
                    partly.to_csv(output_file, index=False)
                    print(f'[Thread {threading.current_thread().name}] Saved {len(links)} links for keyword: {keyword}')
            
            return keyword, links
            
        except Exception as e:
            print(f"Error in thread for keyword {keyword}: {str(e)}")
            return keyword, []
        finally:
            # Always close browser and release semaphore
            if browser:
                self._close_browser(browser)
            self.browser_semaphore.release()
    
    def extract_detail(self, link: str, browser=None):
        """
        Extract detailed information from a specific IEEE paper link.
        
        Args:
            link (str): IEEE paper URL
            browser: Browser instance to use (if None, uses default)
            
        Returns:
            tuple: (title, authors, pdf_link, abstract, doi, submitted_date)
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
            
        page.get(link)
        time.sleep(2)

        # Extract Title
        title = page.ele('css:h1.document-title').text if page.ele('css:h1.document-title') else None
        print(f"[Thread {threading.current_thread().name}] Title:", title)

        # Extract Authors
        authors_elements = page.eles('css:span.authors-info span span')
        authors = [author.text for author in authors_elements]
        print(f"[Thread {threading.current_thread().name}] Authors:", ", ".join(authors))

        # Extract PDF Link
        pdf_link_element = page.ele('css:a.xpl-btn-pdf')
        pdf_link = pdf_link_element.attr('href') if pdf_link_element else None
        if pdf_link == "javascript:void()":
            pdf_link = None
        print(f"[Thread {threading.current_thread().name}] PDF Link:", pdf_link)

        # Extract Abstract
        abstract_div = page.ele('css:div[xplmathjax]')
        abstract = abstract_div.text.replace("Abstract:", "").strip() if abstract_div else None
        print(f"[Thread {threading.current_thread().name}] Abstract length:", len(abstract) if abstract else 0)

        # Extract DOI
        doi_element = page.ele('css:div.stats-document-abstract-doi a')
        doi = f'https://doi.org/{doi_element.text}' if doi_element else None
        print(f"[Thread {threading.current_thread().name}] DOI:", doi)

        # Locate the div containing the conference date
        conf_date_div = page.ele('css:div.doc-abstract-confdate')

        # Extract the submission date
        if conf_date_div:
            conf_date_text = conf_date_div.text  # e.g., "Date of Conference: 13-16 May 2024"
            submitted_date = conf_date_text.split(": ", 1)[1].strip()  # Extracts "13-16 May 2024"
        else:
            submitted_date = None

        # Output the result
        print(f"[Thread {threading.current_thread().name}] Submitted Date:", submitted_date)

        return title, authors, pdf_link, abstract, doi, submitted_date
    
    def _extract_detail_worker(self, link):
        """Worker function for multithreaded detail extraction."""
        # Acquire semaphore to limit concurrent browsers
        self.browser_semaphore.acquire()
        browser = None
        try:
            # Create browser for this thread
            browser = self._create_browser()
            print(f'[Thread {threading.current_thread().name}] Processing link: {link}')
            result = self.extract_detail(link, browser)
            return link, result
        except Exception as e:
            print(f"Error in thread for link {link}: {str(e)}")
            return link, (None, [], None, None, None, None)
        finally:
            # Always close browser and release semaphore
            if browser:
                self._close_browser(browser)
            self.browser_semaphore.release()
    
    def crawl_papers(self, keywords=None, use_multithreading=True):
        """
        Main method to crawl IEEE Xplore for papers based on keywords.
        
        Args:
            keywords (list): List of keywords to search (uses default combinations if None)
            use_multithreading (bool): Whether to use multithreading for crawling
            
        Returns:
            str: Path to the final CSV file containing all papers
        """
        if keywords is None:
            keywords = self.generate_all_combinations()
        
        # Initialize empty lists to store all results
        all_links = []
        
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
                        _, links = future.result()
                        all_links.extend(links)
                        print(f"Completed crawling for keyword: {keyword} ({len(links)} links)")
                    except Exception as e:
                        print(f"Error crawling keyword {keyword}: {str(e)}")
        else:
            # Single-threaded approach (original)
            print("Starting single-threaded crawling...")
            if not self.page:
                self._init_browser()
            
            for keyword in keywords:
                print(f'{keyword} is processed......')

                links = self.crawl_title_link(keyword)

                # Create DataFrame after collecting all data
                partly = pd.DataFrame()
                partly['link'] = links

                if not partly.empty:
                    partly.drop_duplicates(subset=['link'], inplace=True)
                    keyword_filename = keyword.replace('"', '').replace(' ', '_')
                    partly.to_csv(f'{self.output_dir}/crawl_by_{keyword_filename}.csv', index=False)

                all_links.extend(links)

        # Create final DataFrame with all unique links
        full = pd.DataFrame()
        full['link'] = all_links
        full.drop_duplicates(subset=['link'], inplace=True)
        
        output_file = f'{self.output_dir}/all_ieee_papers.csv'
        full.to_csv(output_file, index=False)

        print(f"Total papers collected: {len(full)}")
        print(f"Results saved to: {output_file}")
        
        return output_file
    
    def extract_paper_details(self, csv_file, output_file=None, use_multithreading=True):
        """
        Extract detailed information for all papers in the CSV file.
        
        Args:
            csv_file (str): Path to CSV file containing paper links
            output_file (str): Path to output CSV file (optional)
            use_multithreading (bool): Whether to use multithreading for extraction
            
        Returns:
            pd.DataFrame: DataFrame with detailed paper information
        """
        if output_file is None:
            output_file = csv_file.replace('.csv', '_detailed.csv')
        
        # Read the CSV file with paper links
        papers = pd.read_csv(csv_file)
        
        # Initialize lists to store extracted details
        titles = []
        authors = []
        pdf_links = []
        abstracts = [] 
        dois = []
        submitted_dates = []
        
        if use_multithreading and self.max_threads > 1:
            print(f"Starting multithreaded detail extraction with max {self.max_threads} concurrent browsers...")
            
            # Create a dictionary to maintain order
            link_to_index = {link: idx for idx, link in enumerate(papers['link'])}
            results = [None] * len(papers)
            
            # Use ThreadPoolExecutor for multithreading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all detail extraction tasks
                future_to_link = {
                    executor.submit(self._extract_detail_worker, link): link 
                    for link in papers['link']
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_link):
                    link = future_to_link[future]
                    try:
                        _, result = future.result()
                        idx = link_to_index[link]
                        results[idx] = result
                        print(f"Completed extraction for paper {idx + 1}/{len(papers)}")
                    except Exception as e:
                        print(f"Error extracting details for link {link}: {str(e)}")
                        idx = link_to_index[link]
                        results[idx] = (None, [], None, None, None, None)
            
            # Extract results in order
            for result in results:
                if result:
                    title, author, pdf_link, abstract, doi, submitted_date = result
                    titles.append(title)
                    authors.append(author)
                    pdf_links.append(pdf_link)
                    abstracts.append(abstract)
                    dois.append(doi)
                    submitted_dates.append(submitted_date)
                else:
                    # Handle None results
                    titles.append(None)
                    authors.append([])
                    pdf_links.append(None)
                    abstracts.append(None)
                    dois.append(None)
                    submitted_dates.append(None)
        else:
            # Single-threaded approach (original)
            print("Starting single-threaded detail extraction...")
            if not self.page:
                self._init_browser()
            
            for i, link in enumerate(papers['link']):
                print(f"Processing link {i + 1}/{len(papers)}: {link}")
                
                title, author, pdf_link, abstract, doi, submitted_date = self.extract_detail(link)
                titles.append(title)
                authors.append(author)
                pdf_links.append(pdf_link)
                abstracts.append(abstract)
                dois.append(doi)
                submitted_dates.append(submitted_date)

        # Add extracted details to DataFrame
        papers['title'] = titles
        papers['authors'] = authors
        papers['pdf_link'] = pdf_links
        papers['abstract'] = abstracts
        papers['doi'] = dois
        papers['submitted'] = submitted_dates
        
        # Save the updated DataFrame
        papers.to_csv(output_file, index=False)
        print(f"Detailed paper information saved to: {output_file}")
        
        return papers
    
    def crawl_complete(self, keywords=None, use_multithreading=True):
        """
        Complete crawling workflow: get paper links and extract detailed information.
        
        Args:
            keywords (list): List of keywords to search (uses default combinations if None)
            use_multithreading (bool): Whether to use multithreading for both steps
            
        Returns:
            pd.DataFrame: DataFrame with detailed paper information
        """
        print("Starting IEEE Xplore crawler...")
        
        # Step 1: Crawl paper links
        output_file = self.crawl_papers(keywords, use_multithreading=use_multithreading)
        
        # Step 2: Extract detailed information
        detailed_papers = self.extract_paper_details(output_file, use_multithreading=use_multithreading)
        
        print("IEEE crawling completed successfully!")
        return detailed_papers

def main():
    """Main execution function using the IEEECrawler class."""
    print("🚀 Starting IEEE Crawler with Limited Chrome Instances")
    print("=" * 60)
    
    # Define keyword sets as dictionary - easy to extend with new categories
    keyword_sets = {
        'functional testing': [
             '"text2sql"'] 
        #      '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'
        #  ],
        # 'security': [
        #     '"security"', '"access control"', '"injection"', '"prompt injection"', 
        #     '"defense"', '"attack"', '"vulnerability"'
        # ],
        # 'llm': [
        #     '"llm"', '"large language model"'
        # ],
        # 'ai': [
        #     '"artificial intelligence"', '"machine learning"'
        # ],
        # 'database': [
        #     '"database"', '"sql"', '"query"'
        # ]
    }
    
    # Initialize crawler with keyword sets
    with IEEECrawler(
        headless=False, 
        output_dir='ieee', 
        max_threads=4,
        keyword_sets=keyword_sets
    ) as crawler:
        return crawler.crawl_complete(use_multithreading=True)


if __name__ == "__main__":
    main() 