#!/usr/bin/env python3
"""
Research Pipeline Workflow Examples

This module provides example workflows and usage patterns for the research pipeline.
These examples show how to use the ResearchPipeline class for different scenarios.

Author: Research Team
Date: 2025
"""

from research_pipeline import create_pipeline, ResearchPipelineConfig
import os


def quick_start_example():
    """
    Quick start example - analyze existing papers without crawling.
    Perfect for testing the analysis pipeline.
    """
    print("🚀 Quick Start Example")
    print("=" * 40)
    
    # Create pipeline with API keys
    api_keys = ["your_api_key_here"]  # Replace with actual keys
    pipeline = create_pipeline(api_keys=api_keys)
    
    # Run analysis on existing data (skip crawling)
    results = pipeline.run_full_pipeline(
        skip_crawling=True,      # Use existing crawler outputs
        skip_download=True,      # Skip PDF downloads
        max_papers_to_analyze=5  # Small number for testing
    )
    
    print("✅ Quick start completed!")
    return results


def full_research_workflow():
    """
    Complete research workflow - crawling, combination, and analysis.
    """
    print("🔬 Full Research Workflow")
    print("=" * 40)
    
    # Configure pipeline
    api_keys = ["your_api_key_here"]  # Replace with actual keys
    pipeline = create_pipeline(api_keys=api_keys)
    
    # Step 1: Run selected crawlers
    print("📡 Step 1: Crawling papers...")
    crawl_results = pipeline.run_all_crawlers(['arxiv', 'ieee'])
    
    # Step 2: Combine papers
    print("🔗 Step 2: Combining papers...")
    combined_file = pipeline.combine_papers()
    
    # Step 3: Analyze content
    print("🧠 Step 3: Analyzing content...")
    analysis_results = pipeline.analyze_content(max_papers=20)
    
    # Get summary
    summary = pipeline.get_results_summary()
    print(f"📊 Final Summary: {summary}")
    
    return summary


def custom_keywords_example():
    """
    Example with custom keywords for specific research domains.
    """
    print("🎯 Custom Keywords Example")
    print("=" * 40)
    
    # Define custom keywords for AI testing research
    custom_keywords = {
        'ai_methods': ['"GPT"', '"BERT"', '"Transformer"', '"Neural Network"'],
        'testing_types': ['"Unit Testing"', '"Integration Testing"', '"System Testing"'],
        'domains': ['"web"', '"mobile"', '"desktop"']
    }
    
    # Create pipeline with custom keywords
    pipeline = create_pipeline(keyword_sets=custom_keywords)
    
    # Test crawler combinations
    test_results = pipeline.test_crawler_combinations('arxiv')
    print(f"✅ Generated {test_results.get('total_combinations', 0)} combinations")
    
    return test_results


def analysis_only_workflow():
    """
    Analysis-only workflow for existing CSV files.
    """
    print("🧠 Analysis-Only Workflow")
    print("=" * 40)
    
    # Create pipeline
    api_keys = ["your_api_key_here"]  # Replace with actual keys
    pipeline = create_pipeline(api_keys=api_keys)
    
    # Just run analysis
    analysis_results = pipeline.analyze_content(max_papers=10)
    
    if analysis_results.get('status') == 'success':
        print("✅ Analysis completed!")
        findings = analysis_results.get('findings', {})
        print(f"📊 AI papers: {findings.get('ai_papers', 0)}")
        print(f"🧪 Testing papers: {findings.get('testing_papers', 0)}")
    
    return analysis_results


def configuration_examples():
    """
    Examples of different configuration options.
    """
    print("⚙️ Configuration Examples")
    print("=" * 40)
    
    # Example 1: High-performance configuration
    print("🚀 High-Performance Config:")
    config1 = ResearchPipelineConfig()
    config1.crawl_config['max_threads'] = 8
    config1.crawl_config['headless'] = True
    config1.max_papers_to_analyze = 50
    print(f"   Threads: {config1.crawl_config['max_threads']}")
    print(f"   Headless: {config1.crawl_config['headless']}")
    print(f"   Max analysis: {config1.max_papers_to_analyze}")
    
    # Example 2: Conservative configuration
    print("\n🐌 Conservative Config:")
    config2 = ResearchPipelineConfig()
    config2.crawl_config['max_threads'] = 1
    config2.crawl_config['headless'] = False
    config2.max_papers_to_analyze = 5
    print(f"   Threads: {config2.crawl_config['max_threads']}")
    print(f"   Headless: {config2.crawl_config['headless']}")
    print(f"   Max analysis: {config2.max_papers_to_analyze}")
    
    # Example 3: Recent papers only
    print("\n📅 Recent Papers Config:")
    config3 = ResearchPipelineConfig()
    config3.papers_start_date = "2024-01-01"
    config3.max_papers_to_download = 10
    print(f"   Start date: {config3.papers_start_date}")
    print(f"   Max downloads: {config3.max_papers_to_download}")
    
    return [config1, config2, config3]


def troubleshooting_helper():
    """
    Helper function to diagnose common issues.
    """
    print("🔧 Troubleshooting Helper")
    print("=" * 40)
    
    # Check for common files
    required_files = [
        "output/arxiv/all_arxiv_papers.csv",
        "output/ieee/all_ieee_papers_detailed.csv",
        "filtered_papers/all_papers_consolidated_unique.csv"
    ]
    
    print("📁 Checking for required files:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
        
        if exists:
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                print(f"      📊 Contains {len(df)} rows")
            except Exception as e:
                print(f"      ⚠️ Error reading: {e}")
    
    # Check directories
    required_dirs = [
        "output",
        "filtered_papers",
        "crawl_tools",
        "filter_tools"
    ]
    
    print("\n📂 Checking directories:")
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {dir_path}")
    
    # API key check
    print("\n🔑 API Key Recommendations:")
    print("   • Set API keys in the pipeline configuration")
    print("   • Use environment variable GOOGLE_API_KEY as fallback")
    print("   • Demo mode works without API keys (limited functionality)")
    
    print("\n💡 Common Solutions:")
    print("   • Run crawlers first if no CSV files exist")
    print("   • Use skip_crawling=True to work with existing data")
    print("   • Check file permissions if getting access errors")
    print("   • Reduce max_papers settings if running into limits")


# Command-line interface
if __name__ == "__main__":
    import sys
    
    workflows = {
        'quickstart': quick_start_example,
        'full': full_research_workflow,
        'keywords': custom_keywords_example,
        'analysis': analysis_only_workflow,
        'config': configuration_examples,
        'troubleshoot': troubleshooting_helper
    }
    
    if len(sys.argv) > 1:
        workflow_name = sys.argv[1].lower()
        if workflow_name in workflows:
            workflows[workflow_name]()
        else:
            print(f"❌ Unknown workflow: {workflow_name}")
            print(f"Available workflows: {', '.join(workflows.keys())}")
    else:
        print("🎯 Research Pipeline Workflow Examples")
        print("=" * 50)
        print("Available workflows:")
        for name, func in workflows.items():
            print(f"   python workflow_examples.py {name} - {func.__doc__.split('.')[0].strip()}")
        
        print("\nExample usage:")
        print("   python workflow_examples.py quickstart")
        print("   python workflow_examples.py troubleshoot") 