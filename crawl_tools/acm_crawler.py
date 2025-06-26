"""
ACM Digital Library Paper Crawler

This module provides a class-based approach to crawl ACM Digital Library for academic papers
based on specified keywords and extract links from search results.
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
from urllib.parse import urljoin
import random


class ACMCrawler:
    """
    A class for crawling ACM Digital Library papers based on specified keywords.
    """
    
    def __init__(self, headless=False, output_dir='acm', max_threads=4, keyword_sets=None):
        """
        Initialize the ACM Crawler.
        
        Args:
            headless (bool): Whether to run browser in headless mode
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
            print(f"🔵 Created ACM Chrome instance on port {self.browser.address[1]}")
    
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
        print(f"🔵 Created ACM Chrome instance on port {browser.address[1]} for thread {threading.current_thread().name}")
        return browser
    
    def _close_browser(self, browser):
        """Close and clean up a browser instance."""
        try:
            port = browser.address[1]
            browser.quit()
            print(f"🔴 Closed ACM Chrome instance on port {port} for thread {threading.current_thread().name}")
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
    
    def crawl_links_by_keyword(self, keyword: str, browser=None) -> List[str]:
        """
        Crawl paper links from ACM Digital Library for a given keyword.
        
        Args:
            keyword (str): Search keyword
            browser: Browser instance to use
            
        Returns:
            list: List of paper links
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
        
        links = []
        
        try:
            # Construct search URL
            search_url = f'https://dl.acm.org/action/doSearch?AllField={keyword}'
            print(f"Searching: {search_url}")
            
            page.get(search_url)
            time.sleep(2)
            cnt = 1
            base_url = "https://dl.acm.org"

            while True:
                try:
                    # Wait for page to load with random delay
                    time.sleep(random.randint(2, 5))

                    # Find link elements using CSS selector
                    link_elements = page.eles("css:.issue-item__title a")
                    
                    if not link_elements:
                        print(f"No results found on page {cnt}")
                        break

                    for link in link_elements:
                        relative_link = link.attr("href")
                        # Create full link by combining with base URL
                        full_link = urljoin(base_url, relative_link)
                        if full_link:
                            links.append(full_link)

                    # Check for next page
                    next_button = page.ele("css:a.pagination__btn--next")
                    if not next_button:
                        print(f"Reached last page for keyword: {keyword}")
                        break
                    
                    next_button.click()
                    cnt += 1
                    
                except Exception as e:
                    print(f"Error processing page {cnt}: {str(e)}")
                    break

        except Exception as e:
            print(f"Error searching for keyword {keyword}: {str(e)}")
        
        return links
    
    def _crawl_keyword_worker(self, keyword):
        """Worker function for multithreaded keyword crawling."""
        with self.browser_semaphore:
            browser = self._create_browser()
            try:
                links = self.crawl_links_by_keyword(keyword, browser)
                return keyword, links
            finally:
                self._close_browser(browser)
    
    def save_to_csv(self, data: List, filename: str) -> None:
        """Save data to CSV file."""
        try:
            with self.file_lock:
                if isinstance(data[0], str):
                    df = pd.DataFrame({'link': data})
                else:
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
        Crawl paper links for multiple keywords.
        
        Args:
            keywords (list): List of keywords to search for
            use_multithreading (bool): Whether to use multithreading
            
        Returns:
            dict: Dictionary mapping keywords to their link results
        """
        if keywords is None:
            keywords = self.generate_all_combinations()
        
        all_links = []
        keyword_results = {}
        
        if use_multithreading:
            print(f"Starting ACM crawling with {self.max_threads} threads...")
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_keyword = {
                    executor.submit(self._crawl_keyword_worker, keyword): keyword 
                    for keyword in keywords
                }
                
                for future in as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        original_keyword, links = future.result()
                        keyword_results[original_keyword] = links
                        
                        if links:
                            clean_keyword = keyword.replace('"', '').replace(' ', '_')
                            self.save_to_csv(links, f"crawl_by_{clean_keyword}.csv")
                            all_links.extend(links)
                            
                        print(f"✅ Completed crawling for keyword: {keyword} ({len(links)} links)")
                        
                    except Exception as e:
                        print(f"❌ Error crawling keyword {keyword}: {e}")
        else:
            self._init_browser()
            for keyword in keywords:
                try:
                    links = self.crawl_links_by_keyword(keyword)
                    keyword_results[keyword] = links
                    
                    if links:
                        clean_keyword = keyword.replace('"', '').replace(' ', '_')
                        self.save_to_csv(links, f"crawl_by_{clean_keyword}.csv")
                        all_links.extend(links)
                        
                    print(f"✅ Completed crawling for keyword: {keyword} ({len(links)} links)")
                    
                except Exception as e:
                    print(f"❌ Error crawling keyword {keyword}: {e}")
        
        # Save all links combined
        if all_links:
            self.save_to_csv(all_links, "all_acm_papers.csv")
            print(f"📊 Total links collected: {len(all_links)}")
        
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
        print("🚀 Starting ACM complete crawling pipeline...")
        
        results = self.crawl_papers(keywords, use_multithreading)
        
        print("✅ ACM crawling pipeline completed!")
        return results


def main():
    """Example usage of ACM Crawler."""
    
    keyword_sets = {
        't2sql': ['"text-to-sql"', '"nl2sql"', '"t2sql"', '"text2sql"'] 
        #          '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'],
        # 'security': ['"security"', '"access control"', '"injection"', 
        #             '"prompt injection"', '"defense"', '"attack"', '"vulnerability"'],
        # 'llm': ['"llm"', '"large language model"']
    }
    
    with ACMCrawler(
        headless=False,
        output_dir='acm',
        max_threads=4,
        keyword_sets=keyword_sets
    ) as crawler:
        results = crawler.crawl_complete(use_multithreading=True)
        print(f"🎉 Crawling completed! Results: {len(results)} keyword searches")


if __name__ == "__main__":
    main() 