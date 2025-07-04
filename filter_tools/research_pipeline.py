#!/usr/bin/env python3
"""
Research Paper Pipeline - Main Orchestrator

This module provides a comprehensive pipeline for research paper collection, combination,
filtering, downloading, and analysis. It orchestrates all the tools in the filter_tools
directory to provide a clean, organized workflow.

Features:
- Multi-source paper crawling (ArXiv, IEEE, ACM, Springer, etc.)
- Paper combination and deduplication
- CSV-based content analysis using abstracts
- Configurable workflows and parameters
- Progress tracking and detailed reporting

Author: Research Team
Date: 2025
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import crawlers
from crawl_tools.acm_crawler import ACMCrawler
from crawl_tools.arxiv_crawler import ArxivCrawler
from crawl_tools.ieee_crawler import IEEECrawler
from crawl_tools.mdpi_crawler import MDPICrawler
from crawl_tools.science_direct_crawler import ScienceDirectCrawler
from crawl_tools.springer_crawler import SpringerCrawler

# Import filter tools
from .combine_papers import PaperCombiner
from .download_papers import PaperDownloader
from .content_analysis_csv import CSVContentAnalyzer


class ResearchPipelineConfig:
    """Configuration class for the research pipeline."""
    
    def __init__(self):
        # Directory structure
        self.base_output_dir = "output"
        self.filtered_output_dir = "filtered_papers"
        self.download_dir = os.path.join(self.filtered_output_dir, "downloaded_papers")
        self.analysis_dir = os.path.join(self.filtered_output_dir, "analysis")
        
        # Keywords for crawling
        self.keyword_sets = {
            'functional_testing': [
                '"Functional Testing"', '"Software Testing"', '"Test Case Generation"',
                '"Test Data Generation"', '"Test Automation Frameworks"',
                '"WEB UI (User Interface) Testing"', '"API (Application Programming Interface) Testing"',
                '"Test Oracle Problem"', '"Test Coverage"', '"Test Maintenance"',
                '"Bug Detection"', '"Software Quality Assurance" (SQA)', '"Regression Testing"'
            ],
            'llm': ['"llm"', '"large language model"'],
            'object': ['"web"', '"mobile"', '"code"', '"software"']
        }
        
        # API keys for content analysis
        self.api_keys = []
        
        # Crawler configuration
        self.crawl_config = {
            'headless': False,
            'max_threads': 4,
            'use_multithreading': True
        }
        
        # Paper processing limits
        self.max_papers_to_download = 1000
        self.max_papers_to_analyze = 10
        self.papers_start_date = "2023-01-01"
        
        # File paths
        self.combined_papers_csv = os.path.join(self.filtered_output_dir, "all_papers_consolidated_unique.csv")
        
        # Crawler output directories
        self.crawler_outputs = {
            'IEEE': 'output/ieee/all_ieee_papers_detailed.csv',
            'ArXiv': 'output/arxiv/all_arxiv_papers.csv',
            'ACM': 'output/acm/all_acm_papers.csv',
            'Springer': 'output/springer/all_springer_papers.csv',
            'Science Direct': 'output/science_direct/all_science_direct_papers.csv',
            'MDPI': 'output/mdpi/all_mdpi_papers.csv'
        }
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        for directory in [self.filtered_output_dir, self.download_dir, self.analysis_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def set_api_keys(self, api_keys: List[str]):
        """Set API keys for content analysis."""
        self.api_keys = api_keys
        print(f"🔑 API keys configured: {len(api_keys)} keys available")
    
    def update_keywords(self, keyword_sets: Dict[str, List[str]]):
        """Update keyword sets for crawling."""
        self.keyword_sets = keyword_sets
        print(f"📝 Updated keyword sets: {list(keyword_sets.keys())}")


class ResearchPipeline:
    """Main orchestrator class for the research paper pipeline."""
    
    def __init__(self, config: Optional[ResearchPipelineConfig] = None):
        """
        Initialize the research pipeline.
        
        Args:
            config: Configuration object. If None, creates default config.
        """
        self.config = config if config else ResearchPipelineConfig()
        self.results = {}
        
        print("🚀 Research Pipeline initialized")
        print(f"📁 Output directory: {self.config.filtered_output_dir}")
        print(f"🎯 Keyword groups: {list(self.config.keyword_sets.keys())}")
    
    def get_variable_safely(self, var_name: str, default: Any = None) -> Any:
        """Safely get a variable from results."""
        return self.results.get(var_name, default)
    
    def run_crawler(self, crawler_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific crawler.
        
        Args:
            crawler_name: Name of the crawler ('arxiv', 'ieee', 'acm', etc.)
            **kwargs: Additional arguments for the crawler
            
        Returns:
            Dictionary with crawling results
        """
        print(f"🚀 Starting {crawler_name.upper()} crawling...")
        
        # Map crawler names to classes
        crawler_classes = {
            'arxiv': ArxivCrawler,
            'acm': ACMCrawler,
            'ieee': IEEECrawler,
            'mdpi': MDPICrawler,
            'science_direct': ScienceDirectCrawler,
            'springer': SpringerCrawler
        }
        
        if crawler_name.lower() not in crawler_classes:
            raise ValueError(f"Unknown crawler: {crawler_name}")
        
        crawler_class = crawler_classes[crawler_name.lower()]
        output_dir = f"output/{crawler_name.lower()}"
        
        # Merge configuration with kwargs
        crawler_kwargs = {
            'headless': self.config.crawl_config['headless'],
            'output_dir': output_dir,
            'max_threads': self.config.crawl_config['max_threads'],
            'keyword_sets': self.config.keyword_sets,
            **kwargs
        }
        
        try:
            with crawler_class(**crawler_kwargs) as crawler:
                results = crawler.crawl_complete(
                    use_multithreading=self.config.crawl_config['use_multithreading']
                )
                
                result_info = {
                    'crawler': crawler_name,
                    'results': results,
                    'search_count': len(results),
                    'output_dir': output_dir,
                    'status': 'success'
                }
                
                print(f"🎉 {crawler_name.upper()} crawling completed! Results: {len(results)} keyword searches")
                self.results[f'{crawler_name}_results'] = result_info
                return result_info
                
        except Exception as e:
            error_info = {
                'crawler': crawler_name,
                'error': str(e),
                'status': 'error'
            }
            print(f"❌ Error in {crawler_name.upper()} crawling: {e}")
            self.results[f'{crawler_name}_results'] = error_info
            return error_info
    
    def run_all_crawlers(self, selected_crawlers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all or selected crawlers.
        
        Args:
            selected_crawlers: List of crawler names to run. If None, runs all.
            
        Returns:
            Dictionary with all crawling results
        """
        available_crawlers = ['ieee', 'science_direct', 'springer']
        crawlers_to_run = selected_crawlers if selected_crawlers else available_crawlers
        
        print(f"🎯 Running {len(crawlers_to_run)} crawlers: {', '.join(crawlers_to_run)}")
        
        all_results = {}
        successful_crawls = 0
        
        for crawler_name in crawlers_to_run:
            result = self.run_crawler(crawler_name)
            all_results[crawler_name] = result
            if result.get('status') == 'success':
                successful_crawls += 1
        
        summary = {
            'total_crawlers': len(crawlers_to_run),
            'successful_crawls': successful_crawls,
            'failed_crawls': len(crawlers_to_run) - successful_crawls,
            'individual_results': all_results
        }
        
        print(f"\n📊 Crawling Summary:")
        print(f"   ✅ Successful: {successful_crawls}/{len(crawlers_to_run)}")
        print(f"   ❌ Failed: {summary['failed_crawls']}")
        
        self.results['all_crawler_results'] = summary
        return summary
    
    def combine_papers(self, force_recombine: bool = False) -> Optional[str]:
        """
        Combine papers from all crawler outputs and remove duplicates.
        
        Args:
            force_recombine: Whether to force recombination even if output exists
            
        Returns:
            Path to combined CSV file or None if failed
        """
        print("🔗 Starting paper combination and deduplication...")
        
        # Check if already exists and not forcing recombine
        if os.path.exists(self.config.combined_papers_csv) and not force_recombine:
            print(f"✅ Combined file already exists: {self.config.combined_papers_csv}")
            existing_df = pd.read_csv(self.config.combined_papers_csv)
            print(f"📊 Contains {len(existing_df)} papers")
            self.results['combined_papers_file'] = self.config.combined_papers_csv
            return self.config.combined_papers_csv
        
        # Initialize combiner
        combiner = PaperCombiner()
        
        # Find available input files
        input_files = []
        available_sources = []
        
        for source_name, file_path in self.config.crawler_outputs.items():
            if os.path.exists(file_path):
                input_files.append(file_path)
                available_sources.append(source_name)
                print(f"✅ Found {source_name} papers: {file_path}")
            else:
                print(f"⚠️ {source_name} papers not found: {file_path}")
        
        if len(input_files) == 0:
            print("❌ No input files found for combination")
            return None
        
        print(f"\n🎯 Combining papers from {len(input_files)} sources: {', '.join(available_sources)}")
        
        try:
            output_file = combiner.combine_papers(
                file_paths=input_files,
                output_filename="all_papers_consolidated_unique.csv"
            )
            
            if output_file:
                print(f"\n✅ SUCCESS: Combined papers saved to {output_file}")
                
                # Generate statistics
                combined_df = pd.read_csv(output_file)
                stats = self._generate_combination_stats(combined_df, available_sources)
                
                result_info = {
                    'output_file': output_file,
                    'total_papers': len(combined_df),
                    'sources': available_sources,
                    'stats': stats,
                    'status': 'success'
                }
                
                self.results['combined_papers_file'] = output_file
                self.results['combination_results'] = result_info
                return output_file
            else:
                print("❌ FAILED: Could not combine papers")
                return None
                
        except Exception as e:
            print(f"❌ Error during paper combination: {e}")
            return None
    
    def _generate_combination_stats(self, combined_df: pd.DataFrame, sources: List[str]) -> Dict[str, Any]:
        """Generate statistics for combined papers."""
        print(f"\n📊 Combined Papers Statistics:")
        print(f"- Total unique papers: {len(combined_df)}")
        
        stats = {'total_papers': len(combined_df)}
        
        # Show breakdown by source if possible
        if 'link' in combined_df.columns:
            sources_count = {}
            for _, row in combined_df.iterrows():
                link = str(row['link']).lower()
                if 'ieee' in link:
                    sources_count['IEEE'] = sources_count.get('IEEE', 0) + 1
                elif 'arxiv' in link:
                    sources_count['ArXiv'] = sources_count.get('ArXiv', 0) + 1
                elif 'acm' in link:
                    sources_count['ACM'] = sources_count.get('ACM', 0) + 1
                elif 'springer' in link:
                    sources_count['Springer'] = sources_count.get('Springer', 0) + 1
                elif 'sciencedirect' in link:
                    sources_count['Science Direct'] = sources_count.get('Science Direct', 0) + 1
                elif 'mdpi' in link:
                    sources_count['MDPI'] = sources_count.get('MDPI', 0) + 1
                else:
                    sources_count['Other'] = sources_count.get('Other', 0) + 1
            
            print(f"\n📈 Papers by source:")
            for source, count in sources_count.items():
                percentage = (count / len(combined_df)) * 100
                print(f"- {source}: {count} papers ({percentage:.1f}%)")
            
            stats['source_breakdown'] = sources_count
        
        # Show sample titles
        if 'title' in combined_df.columns and len(combined_df) > 0:
            sample_size = min(5, len(combined_df))
            print(f"\n📝 Sample of {sample_size} paper titles:")
            sample_titles = []
            for i, title in enumerate(combined_df['title'].head(sample_size), 1):
                print(f"{i}. {title}")
                sample_titles.append(title)
            stats['sample_titles'] = sample_titles
        
        return stats
    
    def download_papers(self, max_papers: Optional[int] = None, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Download papers from the combined CSV file.
        
        Args:
            max_papers: Maximum number of papers to download
            start_date: Start date for filtering papers
            
        Returns:
            Dictionary with download results
        """
        print("📥 Starting paper download from consolidated dataset...")
        
        # Get configuration values
        max_papers = max_papers or self.config.max_papers_to_download
        start_date = start_date or self.config.papers_start_date
        
        # Get combined papers file
        combined_file = self.get_variable_safely('combined_papers_file')
        if not combined_file or not os.path.exists(combined_file):
            if os.path.exists(self.config.combined_papers_csv):
                combined_file = self.config.combined_papers_csv
            else:
                print("❌ No consolidated papers file found")
                return {'status': 'error', 'message': 'No input file'}
        
        print(f"📊 Using file: {combined_file}")
        print(f"🎯 Downloading max {max_papers} papers from {start_date}")
        
        try:
            # Initialize downloader
            downloader = PaperDownloader(output_dir=self.config.download_dir)
            
            # Run download pipeline
            filtered_papers, download_results = downloader.run_pipeline(
                input_csv=combined_file,
                start_date=start_date,
                max_papers=max_papers
            )
            
            # Generate results summary
            successful_downloads = sum(1 for status, _ in download_results if status)
            failed_downloads = len(download_results) - successful_downloads
            
            result_info = {
                'total_papers_in_file': len(pd.read_csv(combined_file)),
                'papers_after_filtering': len(filtered_papers),
                'download_attempts': len(download_results),
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'download_results': download_results,
                'status': 'success'
            }
            
            print(f"\n📈 Download Summary:")
            print(f"   📄 Total papers: {result_info['total_papers_in_file']}")
            print(f"   🔍 After filtering: {result_info['papers_after_filtering']}")
            print(f"   📥 Download attempts: {result_info['download_attempts']}")
            print(f"   ✅ Successful: {successful_downloads}")
            print(f"   ❌ Failed: {failed_downloads}")
            
            self.results['download_results'] = result_info
            return result_info
            
        except Exception as e:
            error_info = {'status': 'error', 'message': str(e)}
            print(f"❌ Error during download: {e}")
            self.results['download_results'] = error_info
            return error_info
    
    def download_2023_papers(self, output_subdir: str = "2023_papers") -> Dict[str, Any]:
        """
        Download all papers from 2023 found in the consolidated CSV file.
        
        Args:
            output_subdir: Subdirectory name for saving 2023 papers
            
        Returns:
            Dictionary with download results
        """
        print("🗓️ Starting 2023 papers download from consolidated dataset...")
        
        # Get combined papers file
        combined_file = self.get_variable_safely('combined_papers_file')
        if not combined_file or not os.path.exists(combined_file):
            if os.path.exists(self.config.combined_papers_csv):
                combined_file = self.config.combined_papers_csv
            else:
                print("❌ No consolidated papers file found")
                return {'status': 'error', 'message': 'No input file'}
        
        print(f"📊 Using file: {combined_file}")
        
        try:
            # Initialize downloader
            downloader = PaperDownloader(output_dir=self.config.download_dir)
            
            # Use the specialized 2023 download method
            download_summary = downloader.download_2023_papers(
                input_csv=combined_file,
                output_subdir=output_subdir
            )
            
            # Convert to pipeline format
            result_info = {
                'total_papers_found_2023': download_summary.get('total_found', 0),
                'papers_with_pdfs': download_summary.get('with_pdfs', 0),
                'successful_downloads': download_summary.get('successfully_downloaded', 0),
                'failed_downloads': download_summary.get('failed_downloads', 0),
                'output_directory': download_summary.get('output_directory'),
                'papers_list_csv': download_summary.get('papers_list_csv'),
                'status': 'success' if 'error' not in download_summary else 'error'
            }
            
            if 'error' in download_summary:
                result_info['error_message'] = download_summary['error']
            
            print(f"\n📈 2023 Papers Download Summary:")
            print(f"   📚 Papers from 2023 found: {result_info['total_papers_found_2023']}")
            print(f"   🔗 Papers with PDF links: {result_info['papers_with_pdfs']}")
            print(f"   ✅ Successfully downloaded: {result_info['successful_downloads']}")
            print(f"   ❌ Failed downloads: {result_info['failed_downloads']}")
            print(f"   📁 Output directory: {result_info['output_directory']}")
            
            self.results['download_2023_results'] = result_info
            return result_info
            
        except Exception as e:
            error_info = {'status': 'error', 'message': str(e)}
            print(f"❌ Error during 2023 download: {e}")
            self.results['download_2023_results'] = error_info
            return error_info
    
    def analyze_content(self, max_papers: Optional[int] = None, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Analyze paper content using CSV-based abstract analysis.
        
        Args:
            max_papers: Maximum number of papers to analyze
            force_reanalysis: Whether to force re-analysis even if results exist
            
        Returns:
            Dictionary with analysis results
        """
        print("🧠 Starting CSV-based content analysis...")
        print("=" * 60)
        
        max_papers = max_papers or self.config.max_papers_to_analyze
        analysis_output_dir = os.path.join(self.config.analysis_dir, "csv_analysis")
        
        # Get combined papers file
        combined_file = self.get_variable_safely('combined_papers_file')
        if not combined_file or not os.path.exists(combined_file):
            if os.path.exists(self.config.combined_papers_csv):
                combined_file = self.config.combined_papers_csv
            else:
                print("❌ No consolidated papers file found")
                return {'status': 'error', 'message': 'No input file'}
        
        print(f"📊 Using file: {combined_file}")
        print(f"📁 Analysis output: {analysis_output_dir}")
        
        try:
            # Initialize analyzer
            analyzer = CSVContentAnalyzer()
            
            # Load and inspect CSV
            df = analyzer.load_papers_from_csv(combined_file)
            if df is None:
                return {'status': 'error', 'message': 'Failed to load CSV'}
            
            # Check for abstracts
            has_abstract = df['abstract'].notna() if 'abstract' in df.columns else pd.Series([False] * len(df))
            papers_with_abstracts = has_abstract.sum()
            
            inspection_info = {
                'total_papers': len(df),
                'papers_with_abstracts': papers_with_abstracts,
                'abstract_coverage': papers_with_abstracts / len(df) * 100 if len(df) > 0 else 0
            }
            
            print(f"📋 File inspection:")
            print(f"   📄 Total papers: {inspection_info['total_papers']}")
            print(f"   📝 Papers with abstracts: {inspection_info['papers_with_abstracts']} ({inspection_info['abstract_coverage']:.1f}%)")
            
            # Configure API keys if available
            api_keys_available = False
            if self.config.api_keys:
                analyzer.set_api_keys(self.config.api_keys)
                api_keys_available = True
                print(f"🔑 API keys configured for full analysis")
            else:
                print("⚠️ No API keys - running in demo mode")
            
            # Run analysis
            if api_keys_available and papers_with_abstracts > 0:
                print(f"\n🚀 Running full Gemini API analysis...")
                print(f"🎯 Analyzing {min(max_papers, papers_with_abstracts)} papers")
                
                analysis_results = analyzer.analyze_papers_from_csv(
                    csv_path=combined_file,
                    output_dir=analysis_output_dir,
                    max_papers=max_papers,
                    start_index=0
                )
                
                if analysis_results:
                    summary_file = os.path.join(analysis_output_dir, "analysis_summary.csv")
                    findings = self._process_analysis_findings(summary_file) if os.path.exists(summary_file) else {}
                    
                    result_info = {
                        'analysis_files': analysis_results,
                        'analysis_count': len(analysis_results),
                        'output_directory': analysis_output_dir,
                        'findings': findings,
                        'inspection': inspection_info,
                        'mode': 'full_api',
                        'status': 'success'
                    }
                else:
                    result_info = {
                        'status': 'error',
                        'message': 'API analysis failed',
                        'inspection': inspection_info
                    }
            else:
                # Demo mode
                print(f"\n🔍 Creating demo analysis results...")
                demo_results = self._create_demo_analysis(df, max_papers, analysis_output_dir)
                
                result_info = {
                    'demo_file': demo_results.get('demo_file'),
                    'analysis_count': demo_results.get('analysis_count', 0),
                    'output_directory': analysis_output_dir,
                    'findings': demo_results.get('findings', {}),
                    'inspection': inspection_info,
                    'mode': 'demo',
                    'status': 'success'
                }
            
            self.results['analysis_results'] = result_info
            return result_info
            
        except Exception as e:
            error_info = {'status': 'error', 'message': str(e)}
            print(f"❌ Error during analysis: {e}")
            self.results['analysis_results'] = error_info
            return error_info
    
    def _process_analysis_findings(self, summary_file: str) -> Dict[str, Any]:
        """Process analysis findings from summary file."""
        try:
            summary_df = pd.read_csv(summary_file)
            findings = {}
            
            # Count research domains
            domains = summary_df['research_domain'].dropna()
            if len(domains) > 0:
                domain_counts = {}
                for domain in domains:
                    for keyword in ['testing', 'software', 'AI', 'machine learning', 'LLM', 'neural', 'automation']:
                        if keyword.lower() in domain.lower():
                            domain_counts[keyword] = domain_counts.get(keyword, 0) + 1
                findings['domain_counts'] = domain_counts
            
            # Count AI/LLM papers
            ai_techniques = summary_df['ai_techniques'].dropna()
            ai_count = sum(1 for tech in ai_techniques if tech.strip())
            findings['ai_papers'] = ai_count
            
            # Count testing papers
            testing_domains = summary_df['testing_domain'].dropna()
            testing_count = sum(1 for domain in testing_domains if domain.strip())
            findings['testing_papers'] = testing_count
            
            findings['total_analyzed'] = len(summary_df)
            
            print(f"\n🔍 Key Findings:")
            if findings.get('domain_counts'):
                print(f"   📊 Research domains:")
                for keyword, count in sorted(findings['domain_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      • {keyword}: {count} papers")
            print(f"   🤖 Papers with AI/ML techniques: {findings['ai_papers']}")
            print(f"   🧪 Papers related to testing: {findings['testing_papers']}")
            
            return findings
            
        except Exception as e:
            print(f"⚠️ Error processing findings: {e}")
            return {}
    
    def _create_demo_analysis(self, df: pd.DataFrame, max_papers: int, output_dir: str) -> Dict[str, Any]:
        """Create demo analysis results without API."""
        os.makedirs(output_dir, exist_ok=True)
        
        demo_analysis_data = []
        sample_papers = df.head(max_papers)
        
        analyzer = CSVContentAnalyzer()  # For paper info preparation
        
        for idx, (_, row) in enumerate(sample_papers.iterrows(), 1):
            paper_info = analyzer.prepare_paper_info(row)
            
            # Simple keyword detection
            text_content = str(paper_info.get('title', '') + ' ' + paper_info.get('abstract', '')).lower()
            
            demo_analysis = {
                'title': paper_info.get('title', f'Paper {idx}'),
                'authors': paper_info.get('authors', 'Unknown'),
                'publication_info': paper_info.get('publication_info', 'Unknown'),
                'research_domain': 'Extracted from title/abstract content',
                'problem_statement': paper_info.get('abstract', 'No abstract available')[:200] + "..." if paper_info.get('abstract') else 'No abstract available',
                'ai_techniques': 'LLM' if any(term in text_content for term in ['llm', 'large language', 'gpt', 'ai', 'machine learning']) else 'Not specified',
                'testing_domain': 'Software Testing' if any(term in text_content for term in ['test', 'testing', 'quality', 'bug']) else 'Not specified',
                'primary_contribution': 'Full analysis requires API access'
            }
            demo_analysis_data.append(demo_analysis)
        
        # Save demo results
        demo_summary_df = pd.DataFrame(demo_analysis_data)
        demo_file = os.path.join(output_dir, "demo_analysis_summary.csv")
        demo_summary_df.to_csv(demo_file, index=False)
        
        # Generate demo findings
        ai_papers = sum(1 for item in demo_analysis_data if item['ai_techniques'] != 'Not specified')
        testing_papers = sum(1 for item in demo_analysis_data if item['testing_domain'] != 'Not specified')
        
        findings = {
            'ai_papers': ai_papers,
            'testing_papers': testing_papers,
            'total_analyzed': len(demo_analysis_data)
        }
        
        print(f"📋 Demo analysis saved: {demo_file}")
        print(f"   📊 Analyzed {len(demo_analysis_data)} papers")
        print(f"   🤖 AI/LLM papers: {ai_papers}")
        print(f"   🧪 Testing papers: {testing_papers}")
        
        return {
            'demo_file': demo_file,
            'analysis_count': len(demo_analysis_data),
            'findings': findings
        }
    
    def test_crawler_combinations(self, crawler_name: str = 'arxiv') -> Dict[str, Any]:
        """
        Test if a crawler generates proper keyword combinations.
        
        Args:
            crawler_name: Name of the crawler to test
            
        Returns:
            Dictionary with test results
        """
        print(f"🧪 Testing {crawler_name} crawler combinations...")
        
        # Create test keyword sets
        test_keyword_sets = {
            'functional_testing': ['"Software Testing"', '"Test Case Generation"'],
            'llm': ['"llm"', '"large language model"'],
            'object': ['"web"', '"mobile"', '"code"', '"software"']
        }
        
        try:
            # Map crawler name to class
            crawler_classes = {
                'arxiv': ArxivCrawler,
                'acm': ACMCrawler,
                'ieee': IEEECrawler,
                'mdpi': MDPICrawler,
                'science_direct': ScienceDirectCrawler,
                'springer': SpringerCrawler
            }
            
            if crawler_name.lower() not in crawler_classes:
                return {'status': 'error', 'message': f'Unknown crawler: {crawler_name}'}
            
            crawler_class = crawler_classes[crawler_name.lower()]
            
            # Create test crawler
            test_crawler = crawler_class(
                headless=True,
                output_dir='test_output',
                max_threads=1,
                keyword_sets=test_keyword_sets
            )
            
            # Generate combinations
            combinations = test_crawler.generate_all_combinations(primary_key='llm')
            
            # Verify combinations
            valid_combinations = 0
            for combo in combinations:
                has_llm = any(llm_kw.replace('"', '') in combo for llm_kw in test_keyword_sets['llm'])
                has_testing = any(test_kw.replace('"', '') in combo for test_kw in test_keyword_sets['functional_testing'])
                has_object = any(obj_kw.replace('"', '') in combo for obj_kw in test_keyword_sets['object'])
                
                if has_llm and has_testing and has_object:
                    valid_combinations += 1
            
            success = valid_combinations == len(combinations)
            
            result_info = {
                'crawler': crawler_name,
                'total_combinations': len(combinations),
                'valid_combinations': valid_combinations,
                'success': success,
                'combinations': combinations,
                'status': 'success'
            }
            
            print(f"   📊 Generated {len(combinations)} combinations")
            print(f"   ✅ Valid combinations: {valid_combinations}")
            print(f"   🎯 Success: {'Yes' if success else 'No'}")
            
            return result_info
            
        except Exception as e:
            error_info = {
                'crawler': crawler_name,
                'status': 'error',
                'message': str(e)
            }
            print(f"   ❌ Error: {e}")
            return error_info
    
    def run_full_pipeline(self, selected_crawlers: Optional[List[str]] = None, 
                         skip_crawling: bool = False, skip_download: bool = True,
                         max_papers_to_analyze: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete research pipeline.
        
        Args:
            selected_crawlers: List of crawlers to run
            skip_crawling: Whether to skip the crawling step
            skip_download: Whether to skip the download step
            max_papers_to_analyze: Maximum papers to analyze
            
        Returns:
            Dictionary with complete pipeline results
        """
        print("🚀 Starting Full Research Pipeline")
        print("=" * 50)
        
        pipeline_results = {
            'pipeline_config': {
                'selected_crawlers': selected_crawlers,
                'skip_crawling': skip_crawling,
                'skip_download': skip_download,
                'max_papers_to_analyze': max_papers_to_analyze or self.config.max_papers_to_analyze
            }
        }
        
        # Step 1: Crawling
        if not skip_crawling:
            print("\n📡 Step 1: Running Crawlers")
            crawl_results = self.run_all_crawlers(selected_crawlers)
            pipeline_results['crawling'] = crawl_results
        else:
            print("\n⏭️ Step 1: Skipping crawling (using existing data)")
        
        # Step 2: Combination
        print("\n🔗 Step 2: Combining Papers")
        combination_result = self.combine_papers()
        pipeline_results['combination'] = {
            'output_file': combination_result,
            'status': 'success' if combination_result else 'error'
        }
        
        # Step 3: Download (optional)
        if not skip_download and combination_result:
            print("\n📥 Step 3: Downloading Papers")
            download_results = self.download_papers()
            pipeline_results['download'] = download_results
        else:
            print("\n⏭️ Step 3: Skipping download")
        
        # Step 4: Analysis
        if combination_result:
            print("\n🧠 Step 4: Content Analysis")
            analysis_results = self.analyze_content(max_papers=max_papers_to_analyze)
            pipeline_results['analysis'] = analysis_results
        else:
            print("\n❌ Step 4: Skipping analysis (no combined file)")
        
        # Summary
        print("\n🏁 Pipeline Summary")
        print("=" * 30)
        
        for step, result in pipeline_results.items():
            if step == 'pipeline_config':
                continue
            status = result.get('status', 'unknown') if isinstance(result, dict) else ('success' if result else 'error')
            print(f"{step.capitalize()}: {status}")
        
        self.results['full_pipeline'] = pipeline_results
        return pipeline_results
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all pipeline results."""
        summary = {
            'total_steps_run': len(self.results),
            'steps_completed': [],
            'files_generated': [],
            'statistics': {}
        }
        
        for step, result in self.results.items():
            summary['steps_completed'].append(step)
            
            if isinstance(result, dict):
                # Extract file paths and statistics
                if 'output_file' in result:
                    summary['files_generated'].append(result['output_file'])
                if 'combined_papers_file' in result:
                    summary['files_generated'].append(result)
                
                # Extract key statistics
                if step == 'combination_results':
                    summary['statistics']['total_papers'] = result.get('total_papers', 0)
                elif step == 'analysis_results':
                    summary['statistics']['analyzed_papers'] = result.get('analysis_count', 0)
                elif step == 'download_results':
                    summary['statistics']['downloaded_papers'] = result.get('successful_downloads', 0)
        
        return summary


# Convenience function for quick pipeline setup
def create_pipeline(api_keys: Optional[List[str]] = None, 
                   keyword_sets: Optional[Dict[str, List[str]]] = None) -> ResearchPipeline:
    """
    Create a pre-configured research pipeline.
    
    Args:
        api_keys: List of API keys for content analysis
        keyword_sets: Custom keyword sets for crawling
        
    Returns:
        Configured ResearchPipeline instance
    """
    config = ResearchPipelineConfig()
    
    if api_keys:
        config.set_api_keys(api_keys)
    
    if keyword_sets:
        config.update_keywords(keyword_sets)
    
    return ResearchPipeline(config)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Research Pipeline...")
    
    # Create pipeline with basic configuration
    pipeline = create_pipeline()
    
    # Test combination step (assuming crawler outputs exist)
    result = pipeline.combine_papers()
    if result:
        print(f"✅ Pipeline test successful: {result}")
    else:
        print("⚠️ No crawler outputs found for testing")
    
    # Get summary
    summary = pipeline.get_results_summary()
    print(f"📊 Pipeline summary: {summary}") 