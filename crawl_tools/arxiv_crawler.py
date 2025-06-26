"""
arXiv Paper Crawler

This module provides a class-based approach to crawl arXiv for academic papers
based on specified keywords and extract detailed information from each paper.
"""

from DrissionPage import ChromiumPage, ChromiumOptions
import time
import pandas as pd
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional
import random


class ArxivCrawler:
    """
    A class for crawling arXiv papers based on specified keywords.
    
    This class provides methods to search for papers, extract detailed information,
    and save results to CSV files.
    """
    
    def __init__(self, headless=True, output_dir='arxiv', max_threads=4, keyword_sets=None):
        """
        Initialize the arXiv Crawler.
        
        Args:
            headless (bool): Whether to run browser in headless mode (default True for arXiv)
            output_dir (str): Directory to save output CSV files
            max_threads (int): Maximum number of threads for concurrent crawling
            keyword_sets (dict): Dictionary containing keyword sets
        """
        self.headless = headless
        self.output_dir = output_dir
        self.max_threads = max_threads
        self.browser = None
        self.page = None
        
        # Thread management
        self.file_lock = Lock()
        self.browser_semaphore = threading.Semaphore(max_threads)
        
        # Default keyword sets
        self.keyword_sets = keyword_sets or {
            't2sql': ['"text-to-sql"', '"nl2sql"', '"t2sql"', '"text2sql"', 
                     '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'],
            'security': ['"security"', '"access control"', '"injection"', 
                        '"prompt injection"', '"defense"', '"attack"', '"vulnerability"'],
            'llm': ['"llm"', '"large language model"']
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _init_browser(self):
        """Initialize the main browser instance."""
        if self.browser is None:
            option = ChromiumOptions().auto_port()
            if self.headless:
                option.headless(on_off=True)
            self.browser = ChromiumPage(option)
            self.page = self.browser
            print(f"🔵 Created arXiv Chrome instance on port {self.browser.address[1]}")
    
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
        print(f"🔵 Created arXiv Chrome instance on port {browser.address[1]} for thread {threading.current_thread().name}")
        return browser
    
    def _close_browser(self, browser):
        """Close and clean up a browser instance."""
        try:
            port = browser.address[1]
            browser.quit()
            print(f"🔴 Closed arXiv Chrome instance on port {port} for thread {threading.current_thread().name}")
        except Exception as e:
            print(f"Error closing browser: {e}")
    
    @staticmethod
    def combination_keywords(sets):
        """Generate keyword combination strings from sets."""
        if not sets:
            return []
        combinations = itertools.product(*sets)
        return [' AND '.join(combo) for combo in combinations]
    
    def generate_all_combinations(self, keyword_sets=None, primary_key='t2sql'):
        """Generate all keyword combinations."""
        keyword_sets = keyword_sets or self.keyword_sets
        
        if not keyword_sets or primary_key not in keyword_sets:
            raise ValueError(f"'{primary_key}' keywords must be provided in keyword_sets")
        
        primary_keywords = keyword_sets[primary_key]
        if not primary_keywords:
            raise ValueError(f"'{primary_key}' keyword list cannot be empty")
        
        other_categories = {k: v for k, v in keyword_sets.items() 
                          if k != primary_key and v}
        
        cases = []
        cases.append([primary_keywords])
        
        from itertools import combinations
        category_names = list(other_categories.keys())
        
        for r in range(1, len(category_names) + 1):
            for category_combo in combinations(category_names, r):
                case = [primary_keywords]
                for category in category_combo:
                    case.append(other_categories[category])
                cases.append(case)
        
        all_combinations = []
        for case in cases:
            all_combinations.extend(self.combination_keywords(case))
        
        return all_combinations
    
    def extract_paper_details(self, result) -> Dict:
        """
        Extract details from a single arXiv paper result.
        
        Args:
            result: The DrissionPage element representing the paper result
            
        Returns:
            dict: Dictionary containing the extracted details
        """
        try:
            # Extract arXiv paper link
            link_element = result.ele('css:a[href*="abs"]')
            link = link_element.link if link_element else None

            # Extract PDF link
            pdf_link_element = result.ele('css:a[href*="pdf"]')
            pdf_link = pdf_link_element.link if pdf_link_element else None

            # Extract title
            title_element = result.ele('css:p.title')
            title = title_element.text.strip() if title_element else None

            # Extract authors
            authors_element = result.ele('css:p.authors')
            authors = authors_element.text.replace("Authors:", "").strip() if authors_element else None

            # Extract abstract
            abstract_element = result.ele('css:p.abstract span.abstract-full')
            abstract = abstract_element.text.strip() if abstract_element else None
            if abstract and "△ Less" in abstract:
                abstract = abstract.split("△ Less")[0].strip()

            # Extract submitted date
            submitted_element = result.ele('css:p.is-size-7')
            submitted = submitted_element.text.strip() if submitted_element else None

            return {
                'link': link,
                'pdf_link': pdf_link,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'submitted': submitted
            }

        except Exception as e:
            print(f"Error extracting details: {e}")
            return {}
    
    def crawl_papers_by_keyword(self, keyword: str, browser=None) -> List[Dict]:
        """
        Crawl papers from arXiv for a given keyword.
        
        Args:
            keyword (str): Search keyword
            browser: Browser instance to use
            
        Returns:
            list: List of paper dictionaries
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
        
        papers_data = []
        
        try:
            # Construct search URL
            search_url = f'https://arxiv.org/search/?query={keyword}&searchtype=all&source=header&abstracts=show'
            print(f"Searching: {search_url}")
            
            page.get(search_url)
            cnt = 1
            
            while True:
                # Wait for page to load and scroll to bottom
                page.scroll.to_bottom()
                time.sleep(random.randint(2, 3))
                
                print(f"Page {cnt} loaded, extracting data...")

                # Get arXiv result elements
                arxiv_results = page.eles('css:li.arxiv-result')

                # Extract details from each result
                for result in arxiv_results:
                    try:
                        paper_data = self.extract_paper_details(result)
                        if paper_data:
                            papers_data.append(paper_data)
                    except Exception as e:
                        print(f"Error extracting data from a result: {e}")

                # Check for next page
                next_button = page.ele('css:a.pagination-next')
                if next_button and 'is-invisible' not in next_button.attr('class'):
                    next_button.click()
                    cnt += 1
                    print(f"Loading next page...")
                else:
                    print("No more pages to load.")
                    break

        except Exception as e:
            print(f"Error crawling keyword {keyword}: {e}")
        
        return papers_data
    
    def _crawl_keyword_worker(self, keyword):
        """Worker function for multithreaded keyword crawling."""
        with self.browser_semaphore:
            browser = self._create_browser()
            try:
                papers = self.crawl_papers_by_keyword(keyword, browser)
                return keyword, papers
            finally:
                self._close_browser(browser)
    
    def save_to_csv(self, data: List[Dict], filename: str) -> None:
        """Save data to CSV file."""
        try:
            with self.file_lock:
                df = pd.DataFrame(data)
                if not df.empty:
                    df.drop_duplicates(subset=['link'], inplace=True)
                    filepath = os.path.join(self.output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
    
    def crawl_papers(self, keywords=None, use_multithreading=True):
        """
        Crawl papers for multiple keywords.
        
        Args:
            keywords (list): List of keywords to search for
            use_multithreading (bool): Whether to use multithreading
            
        Returns:
            dict: Dictionary mapping keywords to their paper results
        """
        if keywords is None:
            keywords = self.generate_all_combinations()
        
        all_papers = []
        keyword_results = {}
        
        if use_multithreading:
            print(f"Starting arXiv crawling with {self.max_threads} threads...")
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_keyword = {
                    executor.submit(self._crawl_keyword_worker, keyword): keyword 
                    for keyword in keywords
                }
                
                for future in as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        original_keyword, papers = future.result()
                        keyword_results[original_keyword] = papers
                        
                        if papers:
                            clean_keyword = keyword.replace('"', '').replace(' ', '_')
                            self.save_to_csv(papers, f"crawl_by_{clean_keyword}.csv")
                            all_papers.extend(papers)
                            
                        print(f"✅ Completed crawling for keyword: {keyword} ({len(papers)} papers)")
                        
                    except Exception as e:
                        print(f"❌ Error crawling keyword {keyword}: {e}")
        else:
            self._init_browser()
            for keyword in keywords:
                try:
                    papers = self.crawl_papers_by_keyword(keyword)
                    keyword_results[keyword] = papers
                    
                    if papers:
                        clean_keyword = keyword.replace('"', '').replace(' ', '_')
                        self.save_to_csv(papers, f"crawl_by_{clean_keyword}.csv")
                        all_papers.extend(papers)
                        
                    print(f"✅ Completed crawling for keyword: {keyword} ({len(papers)} papers)")
                    
                except Exception as e:
                    print(f"❌ Error crawling keyword {keyword}: {e}")
        
        # Save all papers combined
        if all_papers:
            self.save_to_csv(all_papers, "all_arxiv_papers.csv")
            print(f"📊 Total papers collected: {len(all_papers)}")
        
        return keyword_results
    
    def crawl_complete(self, keywords=None, use_multithreading=True):
        """
        Complete crawling pipeline.
        
        Args:
            keywords (list): List of keywords to search for
            use_multithreading (bool): Whether to use multithreading
            
        Returns:
            dict: Complete results
        """
        print("🚀 Starting arXiv complete crawling pipeline...")
        
        results = self.crawl_papers(keywords, use_multithreading)
        
        print("✅ arXiv crawling pipeline completed!")
        return results


def main():
    """Example usage of arXiv Crawler."""
    
    keyword_sets = {
        't2sql': ['"text-to-sql"', '"nl2sql"', '"t2sql"', '"text2sql"', 
                 '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'],
        'security': ['"security"', '"access control"', '"injection"', 
                    '"prompt injection"', '"defense"', '"attack"', '"vulnerability"'],
        'llm': ['"llm"', '"large language model"']
    }
    
    with ArxivCrawler(
        headless=False,
        output_dir='arxiv',
        max_threads=2,
        keyword_sets=keyword_sets
    ) as crawler:
        results = crawler.crawl_complete(use_multithreading=True)
        print(f"🎉 Crawling completed! Results: {len(results)} keyword searches")


if __name__ == "__main__":
    main() 