# Filter Tools - Organized Research Pipeline

This directory contains the organized research paper pipeline tools, refactored from the original notebook-based workflow into clean, reusable classes and functions.

## 🏗️ Architecture Overview

### Main Components

#### 1. **ResearchPipeline** (`research_pipeline.py`)
**Main orchestrator class** that coordinates all research activities:

- **ResearchPipelineConfig**: Configuration management
- **ResearchPipeline**: Main pipeline orchestrator
- **create_pipeline()**: Convenience function for quick setup

**Key Features:**
- Multi-source paper crawling (ArXiv, IEEE, ACM, Springer, etc.)
- Automated paper combination and deduplication
- CSV-based abstract analysis using Gemini API
- Configurable workflows and parameters
- Progress tracking and detailed reporting

#### 2. **Individual Tools** (Existing, now integrated)
- `combine_papers.py` - Paper combination and deduplication
- `content_analysis_csv.py` - CSV-based content analysis
- `download_papers.py` - PDF downloading (optional)
- `duplicate_filter.py` - Legacy duplicate filtering
- `keywords_filter_paper.py` - Legacy keyword filtering

#### 3. **Workflow Examples** (`workflow_examples.py`)
Pre-built workflow examples for common use cases:
- Quick start (analysis only)
- Full research workflow
- Custom keywords
- Configuration examples
- Troubleshooting helper

## 🚀 Quick Start

### Basic Usage in Notebook

```python
from filter_tools.research_pipeline import create_pipeline

# Create pipeline with API keys
api_keys = ["your_gemini_api_key"]
pipeline = create_pipeline(api_keys=api_keys)

# Run complete pipeline (skip crawling if data exists)
results = pipeline.run_full_pipeline(
    skip_crawling=True,      # Use existing data
    skip_download=True,      # Skip PDF downloads
    max_papers_to_analyze=10 # Limit for demo
)

# Get results summary
summary = pipeline.get_results_summary()
print(summary)
```

### Individual Steps

```python
# Step 1: Run crawlers
arxiv_result = pipeline.run_crawler('arxiv')
ieee_result = pipeline.run_crawler('ieee')

# Step 2: Combine papers
combined_file = pipeline.combine_papers()

# Step 3: Analyze content
analysis_results = pipeline.analyze_content(max_papers=20)

# Step 4: Download all papers from 2023 (NEW!)
download_summary = pipeline.download_2023_papers()
```

## 📁 File Structure

```
filter_tools/
├── research_pipeline.py      # Main orchestrator class
├── workflow_examples.py      # Example workflows
├── combine_papers.py         # Paper combination tool
├── content_analysis_csv.py   # CSV content analysis
├── download_papers.py        # PDF downloader
├── duplicate_filter.py       # Legacy duplicate filter
├── keywords_filter_paper.py  # Legacy keyword filter
└── README.md                 # This documentation
```

## ⚙️ Configuration Options

### Pipeline Configuration

```python
from filter_tools.research_pipeline import ResearchPipelineConfig

config = ResearchPipelineConfig()

# Customize settings
config.max_papers_to_analyze = 50
config.papers_start_date = "2024-01-01"
config.crawl_config['headless'] = True
config.crawl_config['max_threads'] = 8

# Set API keys
config.set_api_keys(["key1", "key2"])

# Create pipeline with custom config
pipeline = ResearchPipeline(config)
```

### Custom Keywords

```python
custom_keywords = {
    'ai_methods': ['"GPT"', '"BERT"', '"Neural Network"'],
    'testing_types': ['"Unit Testing"', '"Integration Testing"'],
    'domains': ['"web"', '"mobile"', '"API"']
}

pipeline = create_pipeline(keyword_sets=custom_keywords)
```

## 🎯 Workflow Examples

### 1. Analysis-Only Workflow
Perfect for working with existing data:

```python
# Quick analysis of existing papers
pipeline = create_pipeline(api_keys=["your_key"])
results = pipeline.analyze_content(max_papers=10)
```

### 2. Full Research Workflow
Complete pipeline from crawling to analysis:

```python
# Complete workflow
pipeline = create_pipeline(api_keys=["your_key"])

# Run all steps
results = pipeline.run_full_pipeline(
    selected_crawlers=['arxiv', 'ieee'],
    skip_crawling=False,
    skip_download=True,
    max_papers_to_analyze=20
)
```

### 3. Custom Crawler Testing
Test crawler configurations:

```python
# Test individual crawlers
arxiv_test = pipeline.test_crawler_combinations('arxiv')
ieee_test = pipeline.test_crawler_combinations('ieee')
```

## 📊 Output Structure

### Generated Files

```
filtered_papers/
├── all_papers_consolidated_unique.csv  # Combined papers
└── analysis/
    └── csv_analysis/
        ├── analysis_summary.csv        # Summary of all analyses
        ├── 001_Paper_Title.json        # Individual analysis files
        ├── 002_Paper_Title.json
        └── ...
```

### Analysis Output Format

Each paper analysis includes:
- **Paper identification** (title, authors, DOI, publication info)
- **Abstract analysis** (problem statement, methodology, achievements)
- **LLM & AI analysis** (techniques mentioned, specific LLM usage)
- **Software testing analysis** (domains, techniques, challenges)
- **Evaluation & results** (metrics, outcomes)
- **Research contribution** (contributions, significance, limitations)
- **Practical implications** (applications, implementation considerations)

## 🛠️ Usage in Test Notebook

The `test.ipynb` notebook has been completely refactored to use the pipeline:

### Cell Structure
1. **Configuration** - Set up pipeline with API keys
2. **Individual Crawlers** - Run specific crawlers
3. **All Crawlers** - Run multiple crawlers
4. **Paper Combination** - Combine and deduplicate
5. **Content Analysis** - Analyze abstracts
6. **Full Pipeline** - Complete workflow
7. **Testing & Validation** - Test crawler functionality
8. **Results Inspection** - View generated files and statistics
9. **Configuration Management** - Examples of customization

### Migration from Old Notebook

**Before (complex inline code):**
```python
# 50+ lines of complex crawler setup
with ArxivCrawler(headless=..., output_dir=...) as crawler:
    # Complex configuration and error handling
    results = crawler.crawl_complete(...)
# More complex post-processing...
```

**After (clean method calls):**
```python
# Simple, clean method call
arxiv_result = pipeline.run_crawler('arxiv')
```

## 🔧 Troubleshooting

### Common Issues

1. **No crawler outputs found**
   ```python
   # Check existing files
   from filter_tools.workflow_examples import troubleshooting_helper
   troubleshooting_helper()
   ```

2. **API key issues**
   ```python
   # Check API key configuration
   print(f"API keys configured: {len(pipeline.config.api_keys)}")
   ```

3. **Memory issues with large datasets**
   ```python
   # Reduce analysis size
   pipeline.config.max_papers_to_analyze = 5
   ```

### Debug Mode

```python
# Enable verbose output
pipeline.config.crawl_config['headless'] = False  # Show browser
pipeline.config.max_papers_to_analyze = 3         # Small dataset
```

## 📈 Performance Tips

### Optimization Settings

```python
# High-performance configuration
config = ResearchPipelineConfig()
config.crawl_config['max_threads'] = 8      # More threads
config.crawl_config['headless'] = True      # Faster browsing
config.max_papers_to_analyze = 100          # Larger batches
```

### Parallel Processing

```python
# Run multiple crawlers simultaneously
selected_crawlers = ['arxiv', 'ieee', 'acm']
results = pipeline.run_all_crawlers(selected_crawlers)
```

## 🔄 Migration Guide

### From Old Notebook to New Pipeline

1. **Replace imports:**
   ```python
   # Old
   from crawl_tools.arxiv_crawler import ArxivCrawler
   # New  
   from filter_tools.research_pipeline import create_pipeline
   ```

2. **Replace configuration:**
   ```python
   # Old
   CRAWL_CONFIG = {...}
   API_KEYS = [...]
   # New
   pipeline = create_pipeline(api_keys=API_KEYS)
   ```

3. **Replace workflow calls:**
   ```python
   # Old
   with ArxivCrawler(...) as crawler:
       results = crawler.crawl_complete(...)
   # New
   results = pipeline.run_crawler('arxiv')
   ```

### Backwards Compatibility

The original tools (`combine_papers.py`, `content_analysis_csv.py`, etc.) remain available for direct use if needed.

## 📚 API Reference

### Main Classes

- **ResearchPipelineConfig**: Configuration management
- **ResearchPipeline**: Main orchestrator
- **create_pipeline()**: Quick setup function

### Key Methods

- **run_crawler(name)**: Run individual crawler
- **run_all_crawlers()**: Run multiple crawlers  
- **combine_papers()**: Combine and deduplicate papers
- **analyze_content()**: Analyze paper abstracts
- **run_full_pipeline()**: Complete workflow
- **get_results_summary()**: Get pipeline results

### Configuration Options

- **max_papers_to_analyze**: Limit analysis size
- **papers_start_date**: Filter by publication date
- **crawl_config**: Crawler settings (threads, headless mode)
- **api_keys**: Gemini API keys for analysis

### Paper Downloader

The `PaperDownloader` class provides comprehensive PDF downloading capabilities:

```python
from filter_tools.download_papers import PaperDownloader, download_all_2023_papers

# Initialize downloader
downloader = PaperDownloader(output_dir="papers/downloads")

# Standard download with date filtering
filtered_papers, results = downloader.run_pipeline(
    input_csv="consolidated_papers.csv",
    start_date="2024-01-01",
    max_papers=20
)

# NEW: Download all papers from 2023
summary = downloader.download_2023_papers(
    input_csv="consolidated_papers.csv",
    output_subdir="2023_papers"
)

# Convenience function for quick 2023 downloads
summary = download_all_2023_papers(
    output_dir="downloads",
    input_csv="consolidated_papers.csv"
)

# Use with pipeline
pipeline = create_pipeline()
results = pipeline.download_2023_papers(output_subdir="2023_papers")
```

**Features:**
- Date-based filtering with flexible format parsing
- Automatic filename generation from paper titles  
- Progress tracking and download statistics
- Robust error handling for failed downloads
- **NEW**: Specialized 2023 paper batch downloading
- Support for ArXiv, IEEE, Springer, and Science Direct date formats

## 🎉 Benefits of New Structure

### 1. **Simplified Usage**
- Clean method calls instead of complex setup
- Configuration management
- Error handling built-in

### 2. **Better Organization** 
- Logical separation of concerns
- Reusable components
- Consistent interfaces

### 3. **Easier Testing**
- Individual component testing
- Configuration examples
- Troubleshooting helpers

### 4. **Improved Maintenance**
- Centralized configuration
- Consistent error handling
- Better documentation

### 5. **Enhanced Flexibility**
- Custom workflows
- Modular components
- Easy extension

---

## 💡 Next Steps

1. **Test the pipeline** with existing data using `skip_crawling=True`
2. **Customize keywords** for your specific research domain
3. **Configure API keys** for full analysis capabilities
4. **Run full workflows** when ready for new data collection
5. **Extend functionality** by adding new analysis methods

For more examples, see `workflow_examples.py` or run:
```bash
python filter_tools/workflow_examples.py
``` 