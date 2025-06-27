"""
Springer Paper Crawler

This module provides a class-based approach to crawl Springer for academic papers
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


class SpringerCrawler:
    """
    A class for crawling Springer papers based on specified keywords.
    
    This class provides methods to search for papers, extract links,
    and save results to CSV files.
    """
    
    def __init__(self, headless=False, output_dir='springer', max_threads=4, keyword_sets=None):
        """
        Initialize the Springer Crawler.
        
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
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _init_browser(self):
        """Initialize the main browser instance."""
        if self.browser is None:
            option = ChromiumOptions().auto_port()
            if self.headless:
                option.headless(on_off=True)
            self.browser = ChromiumPage(option)
            self.page = self.browser
            print(f"🔵 Created Springer Chrome instance on port {self.browser.address[1]}")
    
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
        print(f"🔵 Created Springer Chrome instance on port {browser.address[1]} for thread {threading.current_thread().name}")
        return browser
    
    def _close_browser(self, browser):
        """Close and clean up a browser instance."""
        try:
            port = browser.address[1]
            browser.quit()
            print(f"🔴 Closed Springer Chrome instance on port {port} for thread {threading.current_thread().name}")
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
    
    def _handle_cookie_dialog(self, page):
        """Handle cookie consent dialog if it appears."""
        try:
            # Wait a bit for the dialog to appear
            time.sleep(2)
            
            # Try to find and click the accept cookies button
            accept_button = page.ele('css:button[data-cc-action="accept"]')
            if accept_button:
                print("🍪 Found cookie dialog, clicking accept...")
                accept_button.click()
                time.sleep(1)
                print("✅ Cookie dialog handled")
            else:
                print("ℹ️ No cookie dialog found")
                
        except Exception as e:
            print(f"⚠️ Error handling cookie dialog: {e}")
    
    def crawl_links_by_keyword(self, keyword: str, browser=None) -> List[str]:
        """
        Crawl paper links from Springer for a given keyword.
        
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
            search_url = f'https://link.springer.com/search?new-search=true&query={keyword}'
            print(f"Searching: {search_url}")
            
            page.get(search_url)
            
            # Handle cookie dialog first
            self._handle_cookie_dialog(page)
            
            time.sleep(2)
            cnt = 1

            while True:
                try:
                    # Wait for page to load
                    time.sleep(5)

                    # Find all paper link elements
                    link_elements = page.eles("css:li.app-card-open a.app-card-open__link")
                    
                    # Extract and build absolute URLs
                    for link in link_elements:
                        href = link.attr('href')
                        if href.startswith('/'):
                            full_url = "https://link.springer.com" + href
                        else:
                            full_url = href  # In case href is already absolute
                        links.append(full_url)

                    # Check for next page
                    next_button = page.ele("css:a.eds-c-pagination__link[rel='next']")
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
    
    def extract_paper_details(self, link: str, browser=None) -> Dict:
        """
        Extract detailed information from a Springer paper link.
        
        Args:
            link (str): Paper URL
            browser: Browser instance to use
            
        Returns:
            dict: Dictionary containing paper details
        """
        page = browser if browser else self.page
        if not page:
            self._init_browser()
            page = self.page
        
        try:
            page.get(link)
            time.sleep(2)

            # Extract PDF link
            pdf_link_element = page.ele("css:a.c-pdf-download__link")
            pdf_link = pdf_link_element.attr('href') if pdf_link_element else None
            
            # Extract title
            title_element = page.ele("css:h1.c-article-title")
            title = title_element.text if title_element else None
            
            # Extract authors
            authors_elements = page.eles("css:a[data-test='author-name']")
            authors = ", ".join([author.text for author in authors_elements]) if authors_elements else None
            
            # Extract abstract
            abstract_element = page.ele("css:div.c-article-section__content p")
            abstract = abstract_element.text if abstract_element else None
            
            # Extract DOI
            doi_element = page.ele("css:span.c-bibliographic-information__value a")
            doi = doi_element.attr('href') if doi_element else None
            
            # Extract publication date
            date_element = page.ele("css:time")
            submitted = date_element.attr('datetime') if date_element else None

            return {
                'link': link,
                'pdf_link': pdf_link,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'doi': doi,
                'submitted': submitted
            }

        except Exception as e:
            print(f"Error extracting details from {link}: {e}")
            return {'link': link}
    
    def _crawl_keyword_worker(self, keyword):
        """Worker function for multithreaded keyword crawling."""
        with self.browser_semaphore:
            browser = self._create_browser()
            try:
                links = self.crawl_links_by_keyword(keyword, browser)
                return keyword, links
            finally:
                self._close_browser(browser)
    
    def save_to_csv(self, data: List, filename: str, columns: List[str] = None) -> None:
        """Save data to CSV file."""
        try:
            with self.file_lock:
                if isinstance(data[0], str):
                    # If data is list of links
                    df = pd.DataFrame({'link': data})
                else:
                    # If data is list of dictionaries
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
            print(f"Starting Springer crawling with {self.max_threads} threads...")
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all keyword crawling tasks
                future_to_keyword = {
                    executor.submit(self._crawl_keyword_worker, keyword): keyword 
                    for keyword in keywords
                }
                
                # Process completed tasks
                for future in as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        original_keyword, links = future.result()
                        keyword_results[original_keyword] = links
                        
                        if links:
                            # Save individual keyword results
                            clean_keyword = keyword.replace('"', '').replace(' ', '_')
                            self.save_to_csv(links, f"crawl_by_{clean_keyword}.csv")
                            all_links.extend(links)
                            
                        print(f"✅ Completed crawling for keyword: {keyword} ({len(links)} links)")
                        
                    except Exception as e:
                        print(f"❌ Error crawling keyword {keyword}: {e}")
        else:
            # Single-threaded crawling
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
            self.save_to_csv(all_links, "all_springer_papers.csv")
            print(f"📊 Total links collected: {len(all_links)}")
        
        return keyword_results
    
    def extract_details_from_links(self, csv_file: str, output_file: str = None, use_multithreading: bool = True):
        """
        Extract detailed information from a CSV file containing paper links.
        
        Args:
            csv_file (str): Path to CSV file containing links
            output_file (str): Output filename for detailed results
            use_multithreading (bool): Whether to use multithreading
        """
        if output_file is None:
            output_file = csv_file.replace('.csv', '_detailed.csv')
        
        # Read links from CSV
        df = pd.read_csv(os.path.join(self.output_dir, csv_file))
        links = df['link'].tolist()
        
        print(f"📖 Extracting details from {len(links)} links...")
        
        if use_multithreading:
            detailed_papers = []
            
            def extract_worker(link):
                with self.browser_semaphore:
                    browser = self._create_browser()
                    try:
                        return self.extract_paper_details(link, browser)
                    finally:
                        self._close_browser(browser)
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_link = {executor.submit(extract_worker, link): link for link in links}
                
                for future in as_completed(future_to_link):
                    link = future_to_link[future]
                    try:
                        details = future.result()
                        detailed_papers.append(details)
                        print(f"✅ Extracted details from: {link}")
                    except Exception as e:
                        print(f"❌ Error extracting details from {link}: {e}")
        else:
            self._init_browser()
            detailed_papers = []
            for link in links:
                try:
                    details = self.extract_paper_details(link)
                    detailed_papers.append(details)
                    print(f"✅ Extracted details from: {link}")
                except Exception as e:
                    print(f"❌ Error extracting details from {link}: {e}")
        
        # Save detailed results
        if detailed_papers:
            self.save_to_csv(detailed_papers, output_file)
            print(f"📊 Detailed extraction completed: {len(detailed_papers)} papers")
    
    def crawl_complete(self, keywords=None, use_multithreading=True, extract_details=False):
        """
        Complete crawling pipeline.
        
        Args:
            keywords (list): List of keywords to search for
            use_multithreading (bool): Whether to use multithreading
            extract_details (bool): Whether to extract detailed information
            
        Returns:
            dict: Complete results
        """
        print("🚀 Starting Springer complete crawling pipeline...")
        
        results = self.crawl_papers(keywords, use_multithreading)
        
        if extract_details:
            print("📖 Starting detail extraction...")
            self.extract_details_from_links("all_springer_papers.csv", "all_springer_papers_detailed.csv", use_multithreading)
        
        print("✅ Springer crawling pipeline completed!")
        return results


def main():
    """Example usage of Springer Crawler."""
    
    keyword_sets = {
        't2sql': ['"text-to-sql"', '"nl2sql"', '"t2sql"', '"text2sql"'] 
        #          '"natural language to sql"', '"semantic parsing to sql"', '"nl to sql"'],
        # 'security': ['"security"', '"access control"', '"injection"', 
        #             '"prompt injection"', '"defense"', '"attack"', '"vulnerability"'],
        # 'llm': ['"llm"', '"large language model"']
    }
    
    with SpringerCrawler(
        headless=False,
        output_dir='springer',
        max_threads=4,
        keyword_sets=keyword_sets
    ) as crawler:
        # Generate combinations and crawl
        results = crawler.crawl_complete(use_multithreading=True, extract_details=True)
        
        print(f"🎉 Crawling completed! Results: {len(results)} keyword searches")


if __name__ == "__main__":
    main() 