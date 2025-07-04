#!/usr/bin/env python3
"""
Content Analysis Tool for Research Papers

This tool analyzes research papers from a consolidated CSV file containing abstracts
and other metadata. It uses Google's Gemini API to perform detailed content analysis
of each paper's abstract and available information.

Features:
- CSV-based paper analysis (no PDF download required)
- Detailed structured analysis using Gemini API  
- Focus on abstracts, titles, and available metadata
- Robust error handling and API key rotation
- JSON output with comprehensive analysis structure

Author: Research Team
Date: 2025
"""

import time
import pandas as pd
import os
import json
import re
from pathlib import Path

class ContentAnalyzer:
    def __init__(self, api_keys=None):
        """
        Initialize the ContentAnalyzer class.
        
        Args:
            api_keys (list): List of API keys for Gemini API
        """
        # Initialize API key management
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.request_count = 0
        self.max_requests_per_key = 10  # Rotate after 10 requests
        self.requests_in_minute = 0
        self.minute_start_time = time.time()
        
        # JSON structure template for abstract-based paper analysis
        self.json_structure = """
    {
        "paper_identification": {
            "title": "Extract the title of the paper exactly as it appears.",
            "authors": "List all authors in the order they are presented, including their affiliations if provided.",
            "publication_venue_year": "Specify the publication venue (e.g., conference, journal) and the year of publication if available.",
            "doi": "Digital Object Identifier if available.",
            "pdf_link": "PDF link if available."
        },
        "abstract_analysis": {
            "problem_statement": {
                "analysis": "Identify and describe the core problem that the paper aims to solve, as explicitly stated in the abstract. Focus on the specific challenge or gap the authors highlight, such as limitations in current approaches, technical challenges, or unresolved issues in the field.",
                "evidence": "Provide a direct quote or precise paraphrase from the abstract that clearly defines the problem."
            },
            "proposed_methodology_summary": {
                "analysis": "Summarize the core methodology, approach, or system proposed by the authors to address the identified problem. Include the key components, techniques, or innovations mentioned in the abstract, such as algorithms, frameworks, models, or methodologies.",
                "evidence": "Quote or specifically paraphrase the part of the abstract that outlines the proposed approach."
            },
            "key_achievements_summary": {
                "analysis": "Summarize the main achievements, key results, or claimed benefits of their approach as highlighted in the abstract. This may include performance improvements, novel capabilities, validation results, or other significant outcomes.",
                "evidence": "Quote or specifically paraphrase the abstract's statements about the results or contributions."
            },
            "research_domain": {
                "analysis": "Identify the specific research domain, field, or application area that this paper focuses on (e.g., software testing, machine learning, natural language processing, mobile app development, etc.).",
                "evidence": "Quote or paraphrase from the title and abstract to support the domain identification."
            },
            "technical_innovation": {
                "analysis": "Describe any technical innovations, novel techniques, or unique approaches mentioned in the abstract. Focus on what makes this work different from existing approaches.",
                "evidence": "Quote or paraphrase the specific innovations mentioned in the abstract."
            },
            "application_context": {
                "analysis": "Identify the specific application context or use case that the paper addresses (e.g., web applications, mobile apps, enterprise software, etc.).",
                "evidence": "Quote or paraphrase from the abstract that describes the application context."
            }
        },
        "llm_and_ai_analysis": {
            "ai_techniques_mentioned": {
                "analysis": "Identify and describe any specific AI or machine learning techniques mentioned in the available content, such as large language models (LLMs), deep learning, neural networks, natural language processing, or other AI approaches. Explain how these techniques are applied or relevant to the research.",
                "evidence": "Quote or specifically paraphrase from the content to support the identification of AI techniques."
            },
            "llm_specific_details": {
                "analysis": "If large language models (LLMs) are mentioned, provide details about which LLMs are used (e.g., GPT, BERT, ChatGPT), how they are applied, any fine-tuning or prompt engineering mentioned, and their role in the proposed solution.",
                "evidence": "Quote or specifically paraphrase the LLM-related content."
            },
            "ai_application_approach": {
                "analysis": "Describe how AI or LLM techniques are integrated into the overall approach or methodology. Include any mentions of training, inference, evaluation, or practical implementation of AI components.",
                "evidence": "Quote or paraphrase the relevant content about AI application."
            }
        },
        "software_testing_analysis": {
            "testing_domain": {
                "analysis": "If the paper relates to software testing, identify the specific testing domain (e.g., unit testing, integration testing, functional testing, test case generation, test automation, bug detection, etc.).",
                "evidence": "Quote or paraphrase from the content that relates to software testing."
            },
            "testing_techniques": {
                "analysis": "Describe any specific testing techniques, methodologies, or tools mentioned in the content (e.g., automated test generation, test oracles, coverage analysis, mutation testing, etc.).",
                "evidence": "Quote or paraphrase the testing techniques mentioned."
            },
            "testing_challenges_addressed": {
                "analysis": "Identify any specific software testing challenges or problems that the paper aims to address (e.g., test maintenance, flaky tests, test coverage, oracle problem, etc.).",
                "evidence": "Quote or paraphrase the testing challenges mentioned in the content."
            }
        },
        "evaluation_and_results": {
            "evaluation_approach": {
                "analysis": "Describe any evaluation methodology, experiments, or validation approaches mentioned in the abstract or available content. Include information about datasets, benchmarks, metrics, or comparison methods used.",
                "evidence": "Quote or paraphrase the evaluation approach from the content."
            },
            "quantitative_results": {
                "analysis": "Extract any specific quantitative results, performance metrics, accuracy scores, improvement percentages, or numerical outcomes mentioned in the abstract.",
                "evidence": "Directly quote or accurately summarize the numerical results mentioned."
            },
            "qualitative_outcomes": {
                "analysis": "Identify any qualitative benefits, improvements, or outcomes claimed in the abstract, such as improved efficiency, better user experience, enhanced reliability, etc.",
                "evidence": "Quote or paraphrase the qualitative outcomes mentioned."
            }
        },
        "research_contribution": {
            "primary_contribution": {
                "analysis": "Identify and describe the primary contribution or main novelty of this research as presented in the available content.",
                "evidence": "Quote or paraphrase the key contribution mentioned."
            },
            "significance_and_impact": {
                "analysis": "Describe any claims about the significance, impact, or importance of this research to the field or practical applications.",
                "evidence": "Quote or paraphrase statements about research significance."
            },
            "limitations_mentioned": {
                "analysis": "Identify any limitations, constraints, or scope boundaries mentioned in the abstract or available content.",
                "evidence": "Quote or paraphrase any limitations mentioned."
            }
        },
        "practical_implications": {
            "real_world_applications": {
                "analysis": "Describe any mentions of real-world applications, practical deployment, or industry relevance of the proposed approach.",
                "evidence": "Quote or paraphrase content related to practical applications."
            },
            "implementation_considerations": {
                "analysis": "Identify any mentions of implementation details, system requirements, scalability considerations, or deployment aspects.",
                "evidence": "Quote or paraphrase implementation-related content."
            }
        }
    }
"""
    
    def set_api_keys(self, api_keys):
        """
        Set the API keys for Gemini API.
        
        Args:
            api_keys (list): List of API keys
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.request_count = 0
    
    def get_current_api_key(self):
        """
        Get the current API key and rotate if needed.
        
        Returns:
            str: Current API key
        """
        if not self.api_keys:
            raise ValueError("No API keys available. Please set API keys using set_api_keys().")
            
        # Check rate limit (15 requests/minute)
        current_time = time.time()
        if current_time - self.minute_start_time >= 60:
            self.requests_in_minute = 0
            self.minute_start_time = current_time
        
        if self.requests_in_minute >= 15:
            print("⏱️ Rate limit reached (15 requests/minute). Waiting 65 seconds...")
            time.sleep(65)
            self.requests_in_minute = 0
            self.minute_start_time = time.time()
        
        # Rotate key after max_requests_per_key requests
        if self.request_count >= self.max_requests_per_key:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.request_count = 0
            print(f"🔄 Switched to API key: {self.api_keys[self.current_key_index][:10]}...")
        
        self.request_count += 1
        self.requests_in_minute += 1
        return self.api_keys[self.current_key_index]
    
    def create_prompt_content(self, paper_info):
        """
        Create the prompt content for the Gemini API using paper information.
        
        Args:
            paper_info (dict): Dictionary containing paper information (title, abstract, authors, etc.)
            
        Returns:
            str: Formatted prompt for the Gemini API
        """
        # Compile available paper information into a coherent text
        paper_content = ""
        
        if paper_info.get('title'):
            paper_content += f"TITLE: {paper_info['title']}\n\n"
            
        if paper_info.get('authors'):
            paper_content += f"AUTHORS: {paper_info['authors']}\n\n"
            
        if paper_info.get('publication_info'):
            paper_content += f"PUBLICATION: {paper_info['publication_info']}\n\n"
            
        if paper_info.get('doi'):
            paper_content += f"DOI: {paper_info['doi']}\n\n"
            
        if paper_info.get('abstract'):
            paper_content += f"ABSTRACT: {paper_info['abstract']}\n\n"
        else:
            paper_content += "ABSTRACT: [No abstract available]\n\n"
            
        if paper_info.get('keywords'):
            paper_content += f"KEYWORDS: {paper_info['keywords']}\n\n"
            
        if paper_info.get('link'):
            paper_content += f"PAPER LINK: {paper_info['link']}\n\n"
            
        if paper_info.get('pdf_link'):
            paper_content += f"PDF LINK: {paper_info['pdf_link']}\n\n"
        
        prompt_string = (
            "You are an expert research paper analyst specializing in Computer Science, Software Engineering, and AI/ML research. "
            "Your primary task is to perform a **thorough, comprehensive, and detailed analysis** of the provided research paper information. "
            "You must generate a detailed report in JSON format based on the available content (primarily title, abstract, and metadata). \n\n"
            
            "**CRITICAL INSTRUCTIONS:**\n"
            "1. **Evidence-Based Analysis**: For EACH analytical point and EVERY field in the JSON structure, you MUST provide direct quotes or highly specific paraphrased evidence extracted directly from the provided paper information.\n"
            "2. **No External Knowledge**: Base your analysis STRICTLY on the content provided. Do not use external knowledge or make inferences beyond what is explicitly stated.\n"
            "3. **JSON Formatting**: Ensure all special characters are properly escaped according to JSON standards. Handle any formatting issues to prevent JSON syntax errors.\n"
            "4. **Comprehensive Coverage**: Address all relevant aspects present in the available information, leaving no significant detail unaddressed.\n"
            "5. **Domain Focus**: Pay special attention to software testing, AI/ML applications, and large language model usage if mentioned.\n\n"
            
            "**When extracting text for inclusion in the JSON:**\n"
            "- Escape all special characters properly (e.g., double quotes with backslash)\n"
            "- Remove or replace problematic characters that could invalidate JSON\n"
            "- Use UTF-8 encoding for non-ASCII characters\n"
            "- Preserve the original meaning and accuracy of extracted evidence\n"
            "- Treat all content as plain strings without interpreting as JSON syntax\n\n"
            
            "**Analysis Focus Areas:**\n"
            "- Identify the research problem and proposed solution\n"
            "- Extract technical details about methodologies and approaches\n"
            "- Highlight any AI/ML or LLM-specific content\n"
            "- Identify software testing related aspects if present\n"
            "- Capture evaluation methods and results\n"
            "- Describe practical implications and applications\n\n"
            
            "The JSON output must **strictly adhere** to the following structure:\n\n" +
            self.json_structure +
            "\n\nProvide your detailed and evidence-backed analysis for the following paper information:\n\n" +
            paper_content +
            "\n\n**Before finalizing your analysis:**\n"
            "1. Review each point to ensure it is directly supported by the provided content\n"
            "2. Verify that the generated JSON is structurally valid and properly formatted\n"
            "3. Ensure all text fields are properly escaped and formatted\n"
            "4. Check that the analysis is comprehensive yet focused on available information\n\n"
            "Your Response (JSON only):\n"
        )

        return prompt_string
    
    def analyze_content_with_gemini(self, paper_info, api_key=None):
        """
        Analyze paper information using the Gemini API.
        
        Args:
            paper_info (dict): Dictionary containing paper information
            api_key (str, optional): API key to use, if None, get from rotation
            
        Returns:
            dict: Parsed JSON result from Gemini API
        """
        try:
            # Import genai here to handle optional dependency
            from google import genai
        except ImportError:
            print("❌ Google GenAI library not found. Please install: pip install google-genai")
            return None
            
        if not api_key:
            api_key = self.get_current_api_key()
            
        client = genai.Client(api_key=api_key)
        prompt_content = self.create_prompt_content(paper_info)
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_content
            )
            result = response.text.strip()
            print("✅ Gemini API request successful")

            # Extract JSON from markdown code block
            match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    json_data = json.loads(json_str)
                    return json_data  # Return parsed dictionary
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON from response: {e}")
                    # Try to find JSON without markdown wrapper
                    try:
                        # Look for JSON patterns in the response
                        json_pattern = r'\{[\s\S]*\}'
                        json_match = re.search(json_pattern, result)
                        if json_match:
                            json_data = json.loads(json_match.group(0))
                            return json_data
                    except:
                        pass
                    return None
            else:
                # Try to parse the entire response as JSON
                try:
                    json_data = json.loads(result)
                    return json_data
                except json.JSONDecodeError:
                    print("❌ No valid JSON found in the response")
                    return None
        except Exception as e:
            print(f"❌ Error calling Gemini API: {e}")
            return None
    
    def load_papers_from_csv(self, csv_path):
        """
        Load papers from the consolidated CSV file.
        
        Args:
            csv_path (str): Path to the consolidated CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing paper information
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 Loaded {len(df)} papers from {csv_path}")
            
            # Check required columns
            required_cols = ['title']
            available_cols = df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in available_cols]
            
            if missing_cols:
                print(f"⚠️ Missing required columns: {missing_cols}")
                print(f"📋 Available columns: {', '.join(available_cols)}")
                return None
            
            print(f"✅ CSV loaded successfully with columns: {', '.join(available_cols)}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV file {csv_path}: {e}")
            return None
    
    def prepare_paper_info(self, row):
        """
        Prepare paper information from a DataFrame row.
        
        Args:
            row: Pandas Series containing paper data
            
        Returns:
            dict: Dictionary with paper information
        """
        paper_info = {}
        
        # Map CSV columns to paper info
        column_mapping = {
            'title': 'title',
            'authors': 'authors', 
            'abstract': 'abstract',
            'doi': 'doi',
            'link': 'link',
            'pdf_link': 'pdf_link',
            'submitted': 'publication_info',
            'keywords': 'keywords'
        }
        
        for csv_col, info_key in column_mapping.items():
            if csv_col in row.index and pd.notna(row[csv_col]):
                paper_info[info_key] = str(row[csv_col]).strip()
        
        return paper_info
    
    def save_analysis_result(self, paper_info, analysis_result, output_dir, index):
        """
        Save analysis result to JSON file.
        
        Args:
            paper_info (dict): Original paper information
            analysis_result (dict): Analysis result from Gemini
            output_dir (str): Output directory
            index (int): Paper index for filename
            
        Returns:
            str: Path to saved JSON file, or None if failed
        """
        try:
            # Create a safe filename from title
            title = paper_info.get('title', f'paper_{index}')
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)  # Remove invalid filename characters
            safe_title = safe_title[:100]  # Limit length
            
            filename = f"{index:03d}_{safe_title}.json"
            output_path = os.path.join(output_dir, filename)
            
            # Combine original info with analysis
            full_result = {
                'original_paper_info': paper_info,
                'analysis': analysis_result,
                'metadata': {
                    'analyzed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_index': index
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, ensure_ascii=False, indent=4)
            
            print(f"💾 Saved analysis to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving analysis result: {e}")
            return None
    
    def analyze_papers_from_csv(self, csv_path, output_dir=None, max_papers=None, start_index=0):
        """
        Analyze papers from a consolidated CSV file.
        
        Args:
            csv_path (str): Path to the consolidated CSV file
            output_dir (str): Output directory for analysis results
            max_papers (int): Maximum number of papers to analyze
            start_index (int): Index to start analysis from
            
        Returns:
            list: List of paths to saved analysis files
        """
        # Load papers from CSV
        df = self.load_papers_from_csv(csv_path)
        if df is None:
            return []
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(csv_path), "analysis_results")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        # Determine papers to analyze
        total_papers = len(df)
        if max_papers:
            end_index = min(start_index + max_papers, total_papers)
        else:
            end_index = total_papers
            
        papers_to_analyze = df.iloc[start_index:end_index]
        print(f"🎯 Analyzing papers {start_index} to {end_index-1} (total: {len(papers_to_analyze)})")
        
        # Check if API keys are available
        if not self.api_keys:
            print("⚠️ No API keys available. Analysis will be limited.")
            return []
        
        # Analyze each paper
        results = []
        successful_analyses = 0
        
        for idx, (_, row) in enumerate(papers_to_analyze.iterrows(), start=start_index):
            try:
                print(f"\n📄 Processing paper {idx + 1}/{end_index}: {row.get('title', 'Untitled')[:100]}...")
                
                # Prepare paper information
                paper_info = self.prepare_paper_info(row)
                
                # Check if we have sufficient information
                if not paper_info.get('title') and not paper_info.get('abstract'):
                    print(f"⚠️ Insufficient information for paper {idx + 1}, skipping...")
                    continue
                
                # Analyze with Gemini
                analysis_result = self.analyze_content_with_gemini(paper_info)
                
                if analysis_result:
                    # Save result
                    saved_path = self.save_analysis_result(
                        paper_info, analysis_result, output_dir, idx + 1
                    )
                    if saved_path:
                        results.append(saved_path)
                        successful_analyses += 1
                        print(f"✅ Successfully analyzed paper {idx + 1}")
                    else:
                        print(f"❌ Failed to save analysis for paper {idx + 1}")
                else:
                    print(f"❌ Failed to analyze paper {idx + 1}")
                
                # Small delay to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing paper {idx + 1}: {e}")
                continue
        
        print(f"\n🎯 Analysis Complete!")
        print(f"   📊 Total papers processed: {len(papers_to_analyze)}")
        print(f"   ✅ Successful analyses: {successful_analyses}")
        print(f"   📁 Results saved to: {output_dir}")
        
        return results
    
    def create_analysis_summary(self, analysis_files, output_dir):
        """
        Create a summary CSV file from analysis results.
        
        Args:
            analysis_files (list): List of paths to analysis JSON files
            output_dir (str): Output directory
            
        Returns:
            str: Path to summary CSV file
        """
        if not analysis_files:
            print("❌ No analysis files to summarize")
            return None
            
        summary_data = []
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                paper_info = data.get('original_paper_info', {})
                analysis = data.get('analysis', {})
                
                # Extract key information for summary
                summary_row = {
                    'title': paper_info.get('title', ''),
                    'authors': paper_info.get('authors', ''),
                    'publication_info': paper_info.get('publication_info', ''),
                    'doi': paper_info.get('doi', ''),
                    'research_domain': analysis.get('abstract_analysis', {}).get('research_domain', {}).get('analysis', ''),
                    'problem_statement': analysis.get('abstract_analysis', {}).get('problem_statement', {}).get('analysis', ''),
                    'methodology': analysis.get('abstract_analysis', {}).get('proposed_methodology_summary', {}).get('analysis', ''),
                    'key_achievements': analysis.get('abstract_analysis', {}).get('key_achievements_summary', {}).get('analysis', ''),
                    'ai_techniques': analysis.get('llm_and_ai_analysis', {}).get('ai_techniques_mentioned', {}).get('analysis', ''),
                    'testing_domain': analysis.get('software_testing_analysis', {}).get('testing_domain', {}).get('analysis', ''),
                    'primary_contribution': analysis.get('research_contribution', {}).get('primary_contribution', {}).get('analysis', ''),
                    'analysis_file': os.path.basename(file_path)
                }
                
                summary_data.append(summary_row)
                
            except Exception as e:
                print(f"⚠️ Error processing {file_path} for summary: {e}")
                continue
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "analysis_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"📋 Summary saved to {summary_path}")
            return summary_path
        else:
            print("❌ No valid data for summary")
            return None


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Content Analyzer with consolidated CSV...")
    
    # Check if consolidated file exists
    consolidated_file = "filtered_papers/all_papers_consolidated_unique.csv"
    
    if os.path.exists(consolidated_file):
        # Initialize analyzer (you would need to provide actual API keys)
        analyzer = ContentAnalyzer(api_keys=["your_api_key_here"])
        
        # Analyze a small sample for testing
        print(f"📄 Analyzing sample papers from {consolidated_file}")
        
        # Test without API keys (just loading and preparation)
        df = analyzer.load_papers_from_csv(consolidated_file)
        if df is not None:
            print(f"✅ Successfully loaded {len(df)} papers")
            
            # Show sample paper info preparation
            sample_row = df.iloc[0]
            paper_info = analyzer.prepare_paper_info(sample_row)
            print(f"\n📋 Sample paper info:")
            for key, value in paper_info.items():
                print(f"   {key}: {value[:100]}...")
        
        print(f"\n💡 To run full analysis with Gemini API:")
        print(f"   1. Set up API keys: analyzer.set_api_keys(['your_key1', 'your_key2'])")
        print(f"   2. Run analysis: analyzer.analyze_papers_from_csv('{consolidated_file}', max_papers=5)")
        
    else:
        print(f"⚠️ Consolidated file not found at: {consolidated_file}")
        print(f"💡 Please run the paper combination tool first to create the consolidated CSV file.")
        print(f"📄 Alternative files to try:")
        for source in ['arxiv', 'ieee', 'acm', 'springer']:
            alt_file = f"output/{source}/all_{source}_papers.csv"
            if os.path.exists(alt_file):
                print(f"   ✅ {alt_file}")
            else:
                print(f"   ❌ {alt_file}")
