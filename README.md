# Crispy Chicken - Academic Paper Crawler

A comprehensive tool for crawling, filtering, analyzing, and downloading academic papers from various sources including arXiv, ACM Digital Library, IEEE Xplore, Science Direct, MDPI, and Springer.

## Features

- Multi-source academic paper crawling
- Keyword-based filtering
- PDF paper downloading
- Content analysis with AI (optional)
- Customizable search parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- Chrome browser (for Selenium-based crawlers)

### Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/crispy-chicken.git
   cd crispy-chicken
   ```

2. Create and activate a virtual environment (recommended):
   ```
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. WebDriver Setup:
   - The project uses webdriver-manager to automatically handle Chrome driver installation
   - Make sure Chrome browser is installed on your system

5. (Optional) Set up API key for content analysis:
   - Create a .env file in the project root directory
   - Add your Google API key for Gemini: `GOOGLE_API_KEY=your_api_key_here`

## Usage

The main workflow is implemented in Jupyter notebooks:

1. Open `main.ipynb` to run the complete pipeline
2. Alternatively, use `test.ipynb` for testing specific components

### Configuration

Key parameters can be adjusted in the notebooks:

- `KEYWORD_SETS`: Define search terms
- `CRAWL_CONFIG`: Configure crawler behavior
- `FILTER_KEYWORDS`: Set filtering criteria
- `SIMILARITY_THRESHOLDS`: Adjust matching sensitivity

## Directory Structure

- `crawl_tools/`: Contains crawler implementations for different academic sources
- `filter_tools/`: Text processing and filtering utilities
- `output/`: Storage for crawled and processed data
- `crawl_results/`: Additional storage for crawler output

## License

[Specify your license here]