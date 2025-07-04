#!/usr/bin/env python3
"""
Quality Assessment Tool for Research Papers

This tool evaluates the quality of research papers based on a structured assessment framework
focusing on research objectives, method descriptions, context, evaluation design, metrics,
baseline comparisons, results presentation, and limitations discussion.

Features:
- PDF-based quality assessment using Google Gemini
- Structured evaluation across 8 quality dimensions
- Detailed scoring with evidence-based justification
- JSON output with comprehensive quality metrics
- Support for batch processing of downloaded papers

Author: Research Team
Date: 2025
"""

import os
import time
import json
import re
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class QualityAssessment:
    def __init__(self, api_keys=None):
        """
        Initialize the QualityAssessment class.
        
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
        
        # Quality assessment JSON structure template
        self.assessment_template = """
{
  "quality_assessment": {
    "QQ1_Research_Objective": {
      "score": "",
      "clarity": "",
      "relevance_to_functional_testing": "",
      "evidence": ""
    },
    "QQ2_Method_Description": {
      "score": "",
      "architecture_diagram": "",
      "stepwise_explanation": "",
      "component_details": "",
      "reproducibility": "",
      "evidence": ""
    },
    "QQ3_Research_Context": {
      "score": "",
      "application_type": "",
      "programming_language": "",
      "frameworks_tools": "",
      "dataset_details": "",
      "evidence": ""
    },
    "QQ4_Evaluation_Design": {
      "score": "",
      "evaluation_type": "",
      "benchmark_selection": "",
      "experiment_procedure": "",
      "repeatability": "",
      "evidence": ""
    },
    "QQ5_Metrics": {
      "score": "",
      "metrics_listed": [],
      "metrics_defined": "",
      "metrics_relevance": "",
      "quantitative_metrics": "",
      "qualitative_metrics": "",
      "evidence": ""
    },
    "QQ6_Baseline_Comparison": {
      "score": "",
      "comparison_type": "",
      "baseline_names": [],
      "baseline_strength": "",
      "statistical_significance": "",
      "evidence": ""
    },
    "QQ7_Results_Presentation": {
      "score": "",
      "results_format": "",
      "clarity": "",
      "conclusion_support": "",
      "key_findings": "",
      "evidence": ""
    },
    "QQ8_LLM_Integration": {
      "score": "",
      "llm_role": "",
      "llm_models_used": [],
      "input_structure": "",
      "output_structure": "",
      "prompt_engineering_details": "",
      "fine_tuning_details": "",
      "llm_limitations_addressed": "",
      "evidence": ""
    },
    "QQ9_Research_Contribution": {
      "score": "",
      "contribution_types": [],
      "novelty_level": "",
      "theoretical_contribution": "",
      "practical_contribution": "",
      "significance_for_field": "",
      "evidence": ""
    },
    "QQ10_Limitations_Discussion": {
      "score": "",
      "limitations_section": "",
      "limitations_categories": {
        "dataset_limitations": [],
        "methodology_limitations": [],
        "llm_specific_limitations": [],
        "generalizability_limitations": [],
        "evaluation_limitations": []
      },
      "limitations_analysis_depth": "",
      "threats_to_validity_categories": {
        "internal_validity": "",
        "external_validity": "",
        "construct_validity": "",
        "conclusion_validity": ""
      },
      "evidence": ""
    },
    "QQ11_Future_Work": {
      "score": "",
      "future_work_section": "",
      "future_work_categories": [],
      "research_directions": [],
      "implementation_suggestions": [],
      "short_term_opportunities": [],
      "long_term_vision": "",
      "evidence": ""
    },
    "QQ12_Input_Output_Definition": {
      "score": "",
      "input_definition_clarity": "",
      "input_formats_described": [],
      "output_definition_clarity": "",
      "output_formats_described": [],
      "transformation_process_clarity": "",
      "examples_provided": "",
      "edge_cases_discussed": "",
      "evidence": ""
    }
  }
}
"""
        
        # Quality assessment criteria explanation
        self.assessment_criteria = """
Quality Assessment Criteria Explanation:

QQ1_Research_Objective:
- Score: "Yes" (clear and explicit), "Partial" (somewhat clear), "No" (unclear/missing)
- Clarity: How clearly the objective is stated (e.g., "Explicit in abstract/introduction")
- Relevance to functional testing: Connection to software testing ("Direct", "Indirect", "None")
- Evidence: Text from paper supporting the assessment

QQ2_Method_Description:
- Score: "Yes" (comprehensive), "Partial" (some details), "No" (insufficient)
- Architecture diagram: Presence and quality of visual representations
- Stepwise explanation: Level of procedural detail
- Component details: Description of individual components
- Reproducibility: Whether the method could be reproduced from description
- Evidence: Text supporting the assessment

QQ3_Research_Context:
- Score: "Yes" (complete), "Partial" (some context), "No" (insufficient)
- Application type: Software context (Web, Mobile, Desktop, API, etc.)
- Programming language: Languages used (Java, Python, etc.)
- Frameworks/tools: Frameworks and tools used (e.g., Spring, Selenium)
- Dataset details: Information about datasets used
- Evidence: Text supporting the assessment

QQ4_Evaluation_Design:
- Score: "Yes" (well-designed), "Partial" (adequate), "No" (insufficient)
- Evaluation type: Type of evaluation (Empirical, Case Study, etc.)
- Benchmark selection: Justification of benchmarks
- Experiment procedure: Detail level of experimental steps
- Repeatability: Whether evaluation could be repeated
- Evidence: Text supporting the assessment

QQ5_Metrics:
- Score: "Yes" (comprehensive), "Partial" (adequate), "No" (insufficient)
- Metrics listed: Specific metrics used (array)
- Metrics defined: How well metrics are defined
- Metrics relevance: Relevance to research questions
- Quantitative metrics: Presence of numerical metrics
- Qualitative metrics: Presence of qualitative assessment
- Evidence: Text supporting the assessment

QQ6_Baseline_Comparison:
- Score: "Yes" (thorough), "Partial" (limited), "No" (absent)
- Comparison type: Type of comparison (e.g., Against state-of-art)
- Baseline names: Names of baseline techniques/approaches (array)
- Baseline strength: Strength of selected baselines
- Statistical significance: Presence of statistical tests
- Evidence: Text supporting the assessment

QQ7_Results_Presentation:
- Score: "Yes" (clear and complete), "Partial" (adequate), "No" (poor)
- Results format: How results are presented (Tables, Figures, Text)
- Clarity: Clarity of result presentation
- Conclusion support: How well conclusions are supported by results
- Key findings: Summary of main results
- Evidence: Text supporting the assessment

QQ8_LLM_Integration:
- Score: "Yes" (comprehensive), "Partial" (adequate), "No" (insufficient/absent)
- LLM role: How LLMs are used in the workflow ("Core component", "Supporting tool", "Evaluation target", etc.)
- LLM models used: Specific models mentioned (GPT-4, BERT, Gemini, etc.)
- Input structure: How inputs to LLMs are structured and formatted
- Output structure: How outputs from LLMs are structured and processed
- Prompt engineering details: Description of prompt design, templates, or engineering techniques
- Fine-tuning details: Information about any fine-tuning or adaptation of models
- LLM limitations addressed: How the paper addresses limitations of LLMs
- Evidence: Text supporting the assessment

QQ9_Research_Contribution:
- Score: "Yes" (significant), "Partial" (modest), "No" (minimal/unclear)
- Contribution types: Nature of contributions ("Methodological", "Empirical", "Theoretical", "Tool", etc.)
- Novelty level: Degree of innovation ("Groundbreaking", "Incremental", "Application of existing techniques")
- Theoretical contribution: Contribution to theory or understanding
- Practical contribution: Real-world applications and implications
- Significance for field: Importance for the research area
- Evidence: Text supporting the assessment

QQ10_Limitations_Discussion:
- Score: "Yes" (thorough), "Partial" (limited), "No" (absent)
- Limitations section: Presence of dedicated limitations section
- Limitations categories: Specific types of limitations discussed:
  - Dataset limitations: Issues with data quality, quantity, diversity, etc.
  - Methodology limitations: Constraints or weaknesses in the approach
  - LLM-specific limitations: Issues related to the use of LLMs
  - Generalizability limitations: Scope constraints of the findings
  - Evaluation limitations: Weaknesses in the evaluation approach
- Limitations analysis depth: How deeply limitations are analyzed
- Threats to validity categories:
  - Internal validity: Factors affecting causal relationships
  - External validity: Generalizability concerns
  - Construct validity: Measurement accuracy concerns
  - Conclusion validity: Statistical inference concerns
- Evidence: Text supporting the assessment

QQ11_Future_Work:
- Score: "Yes" (comprehensive), "Partial" (limited), "No" (absent)
- Future work section: Presence of dedicated future work section
- Future work categories: Types of future work mentioned
- Research directions: Specific research paths suggested
- Implementation suggestions: Concrete implementation ideas
- Short-term opportunities: Immediately actionable research ideas
- Long-term vision: Broader long-term research agenda
- Evidence: Text supporting the assessment

QQ12_Input_Output_Definition:
- Score: "Yes" (clear), "Partial" (somewhat clear), "No" (unclear/missing)
- Input definition clarity: How clearly inputs are defined
- Input formats described: Specific input formats mentioned
- Output definition clarity: How clearly outputs are defined
- Output formats described: Specific output formats mentioned
- Transformation process clarity: How clearly the input-to-output process is described
- Examples provided: Whether examples of inputs/outputs are included
- Edge cases discussed: Whether boundary or exceptional cases are addressed
- Evidence: Text supporting the assessment
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
        logger.info(f"🔑 API keys configured: {len(api_keys)} keys available")
    
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
            logger.info("⏱️ Rate limit reached (15 requests/minute). Waiting 65 seconds...")
            time.sleep(65)
            self.requests_in_minute = 0
            self.minute_start_time = time.time()
        
        # Rotate key after max_requests_per_key requests
        if self.request_count >= self.max_requests_per_key:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.request_count = 0
            logger.info(f"🔄 Switched to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        
        self.request_count += 1
        self.requests_in_minute += 1
        return self.api_keys[self.current_key_index]
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text_content = ""
            # Open the PDF
            with fitz.open(pdf_path) as doc:
                # Get total pages
                total_pages = len(doc)
                logger.info(f"📄 PDF has {total_pages} pages")
                
                # Extract text from all pages, up to a reasonable limit
                max_pages = min(30, total_pages)  # Limit to first 30 pages for efficiency
                for i in range(max_pages):
                    page = doc[i]
                    text_content += page.get_text()
            
            # Limit total text length to 30,000 characters to fit API constraints
            if len(text_content) > 30000:
                logger.info(f"⚠️ Trimming text from {len(text_content)} to 30,000 characters")
                text_content = text_content[:30000]
            
            return text_content
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {e}")
            return None
    
    def extract_paper_metadata(self, pdf_path):
        """
        Extract metadata from PDF file path and file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary with paper metadata
        """
        metadata = {}
        
        try:
            # Extract filename and clean it up to get title
            filename = os.path.basename(pdf_path)
            if filename.endswith('.pdf'):
                filename = filename[:-4]  # Remove .pdf extension
            
            # Clean up filename to make a reasonable title
            title = filename.replace('_', ' ')
            metadata['filename'] = filename
            metadata['title'] = title
            
            # Extract metadata from PDF if possible
            with fitz.open(pdf_path) as doc:
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    if pdf_metadata.get('title'):
                        metadata['title'] = pdf_metadata['title']
                    if pdf_metadata.get('author'):
                        metadata['authors'] = pdf_metadata['author']
                    if pdf_metadata.get('subject'):
                        metadata['subject'] = pdf_metadata['subject']
                    if pdf_metadata.get('keywords'):
                        metadata['keywords'] = pdf_metadata['keywords']
                    if pdf_metadata.get('producer'):
                        metadata['producer'] = pdf_metadata['producer']
            
            return metadata
        except Exception as e:
            logger.error(f"❌ Error extracting metadata from PDF: {e}")
            return {'title': os.path.basename(pdf_path), 'filename': os.path.basename(pdf_path)}
    
    def create_prompt_content(self, paper_text, paper_metadata):
        """
        Create the prompt content for the Gemini API using extracted paper text.
        
        Args:
            paper_text (str): Extracted text from the paper
            paper_metadata (dict): Dictionary containing paper metadata
            
        Returns:
            str: Formatted prompt for the Gemini API
        """
        prompt_string = (
            "You are an expert academic paper reviewer specializing in software engineering and LLM research. "
            "Your task is to perform a structured quality assessment of a research paper according to specific criteria. "
            "Review the provided paper text and generate a detailed quality assessment in JSON format.\n\n"
            
            "**CRITICAL INSTRUCTIONS:**\n"
            "1. **Evidence-Based Assessment**: For EACH criterion, you MUST provide direct quotes or specific "
            "evidence from the paper text to justify your assessment.\n"
            "2. **Complete All Fields**: Fill in ALL fields in the assessment template, including scoring, "
            "specific attributes, and evidence.\n"
            "3. **Use Standard Values**: Use only these standard values for scores: \"Yes\", \"Partial\", \"No\".\n"
            "4. **JSON Validity**: Ensure the output is valid JSON with properly escaped characters.\n"
            "5. **Focus on Quality**: Concentrate on the research quality, methodology, and reporting standards.\n"
            "6. **Special Focus Areas**: Pay particular attention to:\n"
            "   - LLM integration details (how LLMs are used in the workflow)\n"
            "   - Input/output structures and transformations\n"
            "   - Research contributions and novelty\n" 
            "   - Limitations analysis (especially LLM-specific limitations)\n"
            "   - Future work suggestions\n\n"
            
            f"**Paper Information:**\nTitle: {paper_metadata.get('title', 'Unknown')}\n"
        )
        
        if paper_metadata.get('authors'):
            prompt_string += f"Authors: {paper_metadata['authors']}\n"
        
        prompt_string += (
            "\n**Assessment Criteria:**\n" +
            self.assessment_criteria +
            "\n\n**JSON Template:**\n" +
            self.assessment_template +
            "\n\nNow, analyze the following paper text and provide a complete quality assessment "
            "by filling in all fields in the JSON template. Provide specific evidence from the paper "
            "for each aspect of your assessment.\n\n"
            "**Paper Text:**\n" + paper_text[:30000] +
            "\n\nYour Response (JSON only):"
        )
        
        return prompt_string
    
    def assess_quality_with_gemini(self, paper_text, paper_metadata, api_key=None):
        """
        Assess paper quality using the Gemini API.
        
        Args:
            paper_text (str): Extracted text from the paper
            paper_metadata (dict): Dictionary containing paper metadata
            api_key (str, optional): API key to use, if None, get from rotation
            
        Returns:
            dict: Parsed JSON result from Gemini API
        """
        try:
            # Import genai here to handle optional dependency
            from google import genai
        except ImportError:
            logger.error("❌ Google GenAI library not found. Please install: pip install google-genai")
            return None
            
        if not api_key:
            api_key = self.get_current_api_key()
            
        client = genai.Client(api_key=api_key)
        prompt_content = self.create_prompt_content(paper_text, paper_metadata)
        
        try:
            logger.info(f"🧠 Requesting quality assessment from Gemini API...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',  # Use the 'flash' model for faster, more efficient processing
                contents=prompt_content
            )
            result = response.text.strip()
            logger.info("✅ Gemini API request successful")

            # Extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', result, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    json_data = json.loads(json_str)
                    return json_data  # Return parsed dictionary
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON from response: {e}")
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
                    logger.error("❌ No valid JSON found in the response")
                    return None
        except Exception as e:
            logger.error(f"❌ Error calling Gemini API: {e}")
            return None
    
    def save_assessment_result(self, pdf_path, paper_metadata, assessment_result, output_dir):
        """
        Save assessment result to JSON file.
        
        Args:
            pdf_path (str): Path to the original PDF
            paper_metadata (dict): Original paper metadata
            assessment_result (dict): Assessment result from Gemini
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved JSON file, or None if failed
        """
        try:
            # Create a safe filename from title
            title = paper_metadata.get('title', os.path.basename(pdf_path))
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)  # Remove invalid filename characters
            safe_title = safe_title[:100]  # Limit length
            
            filename = f"quality_assessment_{safe_title}.json"
            output_path = os.path.join(output_dir, filename)
            
            # Combine original info with assessment
            full_result = {
                'paper_metadata': paper_metadata,
                'pdf_path': pdf_path,
                'assessment': assessment_result,
                'metadata': {
                    'assessed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved assessment to {os.path.basename(output_path)}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error saving assessment result: {e}")
            return None
    
    def assess_paper(self, pdf_path, output_dir=None):
        """
        Assess quality of a single paper.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str, optional): Output directory for assessment results
            
        Returns:
            dict: Assessment result
        """
        try:
            # Extract text from PDF
            logger.info(f"📄 Processing paper: {os.path.basename(pdf_path)}")
            paper_text = self.extract_text_from_pdf(pdf_path)
            
            if not paper_text:
                logger.error(f"❌ Failed to extract text from {pdf_path}")
                return None
                
            # Extract paper metadata
            paper_metadata = self.extract_paper_metadata(pdf_path)
            
            # Check if API keys are available
            if not self.api_keys:
                logger.error("⚠️ No API keys available. Cannot perform assessment.")
                return None
                
            # Assess paper quality
            assessment_result = self.assess_quality_with_gemini(paper_text, paper_metadata)
            
            if not assessment_result:
                logger.error(f"❌ Failed to get quality assessment for {pdf_path}")
                return None
                
            # Create output directory if needed
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(pdf_path), "quality_assessments")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save result
            output_path = self.save_assessment_result(
                pdf_path=pdf_path,
                paper_metadata=paper_metadata,
                assessment_result=assessment_result,
                output_dir=output_dir
            )
            
            if output_path:
                logger.info(f"✅ Quality assessment completed and saved to {output_path}")
                return assessment_result
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error assessing paper {pdf_path}: {e}")
            return None
    
    def batch_assess_papers(self, pdf_dir, output_dir=None, max_papers=None, file_pattern='*.pdf'):
        """
        Assess quality of multiple papers in a directory.
        
        Args:
            pdf_dir (str): Directory containing PDF files
            output_dir (str, optional): Output directory for assessment results
            max_papers (int, optional): Maximum number of papers to assess
            file_pattern (str): File pattern to match PDF files
            
        Returns:
            dict: Summary of assessment results
        """
        logger.info(f"🧠 Starting Batch Quality Assessment")
        logger.info("=" * 50)
        
        # Create output directory
        if not output_dir:
            output_dir = os.path.join(pdf_dir, "quality_assessments")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Find all PDF files
        pdf_files = list(Path(pdf_dir).glob(file_pattern))
        logger.info(f"📊 Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        if max_papers and max_papers < len(pdf_files):
            pdf_files = pdf_files[:max_papers]
            logger.info(f"🎯 Limiting assessment to {max_papers} papers")
        
        # Check if API keys are available
        if not self.api_keys:
            logger.error("⚠️ No API keys available. Cannot perform assessment.")
            logger.info("💡 Use assessor.set_api_keys(['your_key1', 'your_key2']) to add API keys")
            return {
                'status': 'error',
                'message': 'No API keys available',
                'total_papers': len(pdf_files),
                'successful_assessments': 0,
                'failed_assessments': len(pdf_files)
            }
        
        # Assess each paper
        results = []
        successful_assessments = 0
        failed_assessments = 0
        
        # Create a CSV for summary data
        summary_data = []
        
        for pdf_file in tqdm(pdf_files, desc="Assessing papers"):
            try:
                assessment_result = self.assess_paper(str(pdf_file), output_dir)
                
                if assessment_result:
                    results.append({
                        'pdf_path': str(pdf_file),
                        'output_file': os.path.join(output_dir, f"quality_assessment_{os.path.basename(pdf_file)[:-4]}.json"),
                        'status': 'success'
                    })
                    successful_assessments += 1
                    
                    # Add to summary data
                    try:
                        qa = assessment_result.get('quality_assessment', {})
                        
                        # Function to join list fields with commas
                        def join_list(lst):
                            if isinstance(lst, list):
                                return ", ".join(str(x) for x in lst if x)
                            return ""
                        
                        # Calculate scores for this paper
                        paper_scores = self.calculate_paper_score({'assessment': {'quality_assessment': qa}})
                        
                        # Extract scores
                        overall_score = paper_scores.get('percentage_score', 0.0)
                        letter_grade = paper_scores.get('letter_grade', 'N/A')
                        
                        # Get category scores
                        category_scores = paper_scores.get('category_scores', {})
                        research_basics = category_scores.get('research_basics', {}).get('percentage', 0.0)
                        methodology = category_scores.get('methodology', {}).get('percentage', 0.0)
                        evaluation = category_scores.get('evaluation', {}).get('percentage', 0.0)
                        llm_impl = category_scores.get('llm_implementation', {}).get('percentage', 0.0)
                        critical_analysis = category_scores.get('critical_analysis', {}).get('percentage', 0.0)
                        
                        summary_data.append({
                            'filename': os.path.basename(pdf_file),
                            
                            # Overall scores
                            'overall_score': round(overall_score, 1),
                            'letter_grade': letter_grade,
                            
                            # Category scores
                            'research_basics_score': round(research_basics, 1),
                            'methodology_score': round(methodology, 1),
                            'evaluation_score': round(evaluation, 1),
                            'llm_implementation_score': round(llm_impl, 1),
                            'critical_analysis_score': round(critical_analysis, 1),
                            
                            # Individual criterion scores
                            'research_objective_score': qa.get('QQ1_Research_Objective', {}).get('score', 'N/A'),
                            'method_description_score': qa.get('QQ2_Method_Description', {}).get('score', 'N/A'),
                            'research_context_score': qa.get('QQ3_Research_Context', {}).get('score', 'N/A'),
                            'evaluation_design_score': qa.get('QQ4_Evaluation_Design', {}).get('score', 'N/A'),
                            'metrics_score': qa.get('QQ5_Metrics', {}).get('score', 'N/A'),
                            'baseline_comparison_score': qa.get('QQ6_Baseline_Comparison', {}).get('score', 'N/A'),
                            'results_presentation_score': qa.get('QQ7_Results_Presentation', {}).get('score', 'N/A'),
                            'llm_integration_score': qa.get('QQ8_LLM_Integration', {}).get('score', 'N/A'),
                            'research_contribution_score': qa.get('QQ9_Research_Contribution', {}).get('score', 'N/A'),
                            'limitations_discussion_score': qa.get('QQ10_Limitations_Discussion', {}).get('score', 'N/A'),
                            'future_work_score': qa.get('QQ11_Future_Work', {}).get('score', 'N/A'),
                            'input_output_definition_score': qa.get('QQ12_Input_Output_Definition', {}).get('score', 'N/A'),
                            
                            # Additional LLM integration details
                            'llm_role': qa.get('QQ8_LLM_Integration', {}).get('llm_role', 'N/A'),
                            'llm_models_used': join_list(qa.get('QQ8_LLM_Integration', {}).get('llm_models_used', [])),
                            
                            # Research contribution details
                            'contribution_types': join_list(qa.get('QQ9_Research_Contribution', {}).get('contribution_types', [])),
                            'novelty_level': qa.get('QQ9_Research_Contribution', {}).get('novelty_level', 'N/A'),
                            
                            # Limitations discussion details
                            'limitations_depth': qa.get('QQ10_Limitations_Discussion', {}).get('limitations_analysis_depth', 'N/A'),
                            
                            # Future work details
                            'research_directions': join_list(qa.get('QQ11_Future_Work', {}).get('research_directions', [])),
                            
                            # Input/output details
                            'input_clarity': qa.get('QQ12_Input_Output_Definition', {}).get('input_definition_clarity', 'N/A'),
                            'output_clarity': qa.get('QQ12_Input_Output_Definition', {}).get('output_definition_clarity', 'N/A')
                        })
                    except:
                        pass
                else:
                    results.append({
                        'pdf_path': str(pdf_file),
                        'status': 'error'
                    })
                    failed_assessments += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file}: {e}")
                results.append({
                    'pdf_path': str(pdf_file),
                    'status': 'error',
                    'message': str(e)
                })
                failed_assessments += 1
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, "quality_assessment_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"📊 Saved assessment summary to {summary_csv_path}")
        
        # Create summary
        summary = {
            'status': 'success',
            'total_papers': len(pdf_files),
            'successful_assessments': successful_assessments,
            'failed_assessments': failed_assessments,
            'output_directory': output_dir,
            'summary_csv': os.path.join(output_dir, "quality_assessment_summary.csv") if summary_data else None,
            'assessment_details': results
        }
        
        # Calculate scores for the summary data
        if summary_data:
            overall_scores = [item.get('overall_score', 0) for item in summary_data]
            avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
            
            # Count letter grades
            letter_grades = {}
            for item in summary_data:
                grade = item.get('letter_grade', 'N/A')
                if grade not in letter_grades:
                    letter_grades[grade] = 0
                letter_grades[grade] += 1
            
            # Format letter grade distribution
            grade_dist = ", ".join([f"{grade}: {count}" for grade, count in sorted(letter_grades.items())])
            
            logger.info("=" * 50)
            logger.info(f"📊 Assessment Summary:")
            logger.info(f"   📄 Total papers: {len(pdf_files)}")
            logger.info(f"   ✅ Successful assessments: {successful_assessments}")
            logger.info(f"   ❌ Failed assessments: {failed_assessments}")
            logger.info(f"   🎓 Average overall score: {avg_score:.1f}")
            logger.info(f"   🏆 Grade distribution: {grade_dist}")
            logger.info(f"   📁 Output directory: {output_dir}")
        else:
            logger.info("=" * 50)
            logger.info(f"📊 Assessment Summary:")
            logger.info(f"   📄 Total papers: {len(pdf_files)}")
            logger.info(f"   ✅ Successful assessments: {successful_assessments}")
            logger.info(f"   ❌ Failed assessments: {failed_assessments}")
            logger.info(f"   📁 Output directory: {output_dir}")
        
        return summary
    
    def assess_2023_papers(self, papers_dir="filtered_papers/downloaded_papers/2023_papers", output_dir=None, max_papers=None):
        """
        Assess quality of papers from 2023.
        
        Args:
            papers_dir (str): Directory containing 2023 papers
            output_dir (str, optional): Output directory for assessment results
            max_papers (int, optional): Maximum number of papers to assess
            
        Returns:
            dict: Summary of assessment results
        """
        logger.info(f"🧠 Starting 2023 Papers Quality Assessment")
        logger.info("=" * 50)
        
        # Create output directory if not provided
        if not output_dir:
            output_dir = os.path.join(papers_dir, "quality_assessments")
        
        return self.batch_assess_papers(
            pdf_dir=papers_dir,
            output_dir=output_dir,
            max_papers=max_papers
        )
    
    def calculate_paper_score(self, assessment):
        """
        Calculate numerical score for a paper based on quality assessment.
        
        Args:
            assessment (dict): Paper assessment dictionary
            
        Returns:
            dict: Dictionary with score details
        """
        # Score weights for each criterion
        weights = {
            'QQ1_Research_Objective': 1.0,
            'QQ2_Method_Description': 1.0,
            'QQ3_Research_Context': 0.8,
            'QQ4_Evaluation_Design': 1.0,
            'QQ5_Metrics': 0.9,
            'QQ6_Baseline_Comparison': 0.8,
            'QQ7_Results_Presentation': 0.8,
            'QQ8_LLM_Integration': 1.2,  # Higher weight for LLM integration
            'QQ9_Research_Contribution': 1.1,  # Higher weight for research contribution
            'QQ10_Limitations_Discussion': 1.0,
            'QQ11_Future_Work': 0.8,
            'QQ12_Input_Output_Definition': 1.0
        }
        
        # Score values for Yes/Partial/No
        score_values = {
            'Yes': 1.0,
            'Partial': 0.5,
            'No': 0.0,
            # Handle any other values gracefully
            'N/A': 0.0,
            None: 0.0
        }
        
        qa = assessment.get('assessment', {}).get('quality_assessment', {})
        
        # Calculate scores for each criterion
        criterion_scores = {}
        total_score = 0.0
        total_weight = 0.0
        max_possible_score = 0.0
        
        for criterion, weight in weights.items():
            score_text = qa.get(criterion, {}).get('score')
            score_value = score_values.get(score_text, 0.0)
            weighted_score = score_value * weight
            
            criterion_scores[criterion] = {
                'raw_score': score_value,
                'weight': weight,
                'weighted_score': weighted_score,
                'score_text': score_text
            }
            
            total_score += weighted_score
            total_weight += weight
            max_possible_score += weight
        
        # Calculate overall score as percentage
        percentage_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        # Letter grade based on percentage
        letter_grade = 'A+'
        if percentage_score < 97: letter_grade = 'A'
        if percentage_score < 93: letter_grade = 'A-'
        if percentage_score < 90: letter_grade = 'B+'
        if percentage_score < 87: letter_grade = 'B'
        if percentage_score < 83: letter_grade = 'B-'
        if percentage_score < 80: letter_grade = 'C+'
        if percentage_score < 77: letter_grade = 'C'
        if percentage_score < 73: letter_grade = 'C-'
        if percentage_score < 70: letter_grade = 'D+'
        if percentage_score < 67: letter_grade = 'D'
        if percentage_score < 63: letter_grade = 'D-'
        if percentage_score < 60: letter_grade = 'F'
        
        # Calculate category scores
        categories = {
            'research_basics': ['QQ1_Research_Objective', 'QQ3_Research_Context', 'QQ9_Research_Contribution'],
            'methodology': ['QQ2_Method_Description', 'QQ4_Evaluation_Design', 'QQ12_Input_Output_Definition'],
            'evaluation': ['QQ5_Metrics', 'QQ6_Baseline_Comparison', 'QQ7_Results_Presentation'],
            'llm_implementation': ['QQ8_LLM_Integration'],
            'critical_analysis': ['QQ10_Limitations_Discussion', 'QQ11_Future_Work']
        }
        
        category_scores = {}
        for category_name, category_criteria in categories.items():
            category_total = 0.0
            category_weight = 0.0
            category_max = 0.0
            
            for criterion in category_criteria:
                if criterion in criterion_scores:
                    category_total += criterion_scores[criterion]['weighted_score']
                    category_weight += weights[criterion]
                    category_max += weights[criterion]
            
            if category_max > 0:
                category_scores[category_name] = {
                    'score': category_total,
                    'weight': category_weight,
                    'percentage': (category_total / category_max) * 100,
                    'max_score': category_max
                }
        
        return {
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'percentage_score': percentage_score,
            'letter_grade': letter_grade,
            'criterion_scores': criterion_scores,
            'category_scores': category_scores
        }
    
    def generate_score_visualization(self, assessments, output_dir):
        """
        Generate visualizations of the quality assessment scores.
        
        Args:
            assessments (list): List of assessment dictionaries
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: Paths to generated visualization files
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.error("❌ matplotlib is required for generating visualizations. Install with: pip install matplotlib")
            return []
        
        visualization_paths = []
        
        # Extract scores
        papers_data = []
        for assessment in assessments:
            paper_metadata = assessment.get('paper_metadata', {})
            paper_scores = assessment.get('score', {})
            qa = assessment.get('assessment', {}).get('quality_assessment', {})
            
            if not paper_scores:
                continue
            
            title = paper_metadata.get('title', 'Unknown Paper')
            short_title = title[:30] + '...' if len(title) > 30 else title
            
            # Get category scores
            category_scores = paper_scores.get('category_scores', {})
            
            papers_data.append({
                'title': short_title,
                'overall_score': paper_scores.get('percentage_score', 0.0),
                'letter_grade': paper_scores.get('letter_grade', 'F'),
                'research_basics': category_scores.get('research_basics', {}).get('percentage', 0.0),
                'methodology': category_scores.get('methodology', {}).get('percentage', 0.0),
                'evaluation': category_scores.get('evaluation', {}).get('percentage', 0.0),
                'llm_implementation': category_scores.get('llm_implementation', {}).get('percentage', 0.0),
                'critical_analysis': category_scores.get('critical_analysis', {}).get('percentage', 0.0)
            })
        
        if not papers_data:
            logger.error("❌ No score data available for visualization")
            return []
        
        # Sort papers by overall score
        papers_data = sorted(papers_data, key=lambda x: x['overall_score'], reverse=True)
        
        # 1. Create overall score chart
        try:
            plt.figure(figsize=(10, 6))
            
            titles = [p['title'] for p in papers_data]
            scores = [p['overall_score'] for p in papers_data]
            grades = [p['letter_grade'] for p in papers_data]
            
            # Define colors based on letter grades
            grade_colors = {
                'A+': '#1a9641', 'A': '#1a9641', 'A-': '#1a9641',
                'B+': '#a6d96a', 'B': '#a6d96a', 'B-': '#a6d96a',
                'C+': '#ffffbf', 'C': '#ffffbf', 'C-': '#ffffbf',
                'D+': '#fdae61', 'D': '#fdae61', 'D-': '#fdae61',
                'F': '#d7191c'
            }
            
            # Apply colors based on grades
            colors = [grade_colors.get(grade, '#d7191c') for grade in grades]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(titles))
            plt.barh(y_pos, scores, color=colors)
            plt.yticks(y_pos, titles)
            plt.xlabel('Score (%)')
            plt.title('Overall Quality Scores by Paper')
            
            # Add grade labels to the end of each bar
            for i, (score, grade) in enumerate(zip(scores, grades)):
                plt.text(score + 1, i, grade, va='center')
            
            plt.xlim(0, 105)  # Leave space for grade labels
            plt.tight_layout()
            
            # Save figure
            overall_chart_path = os.path.join(output_dir, 'overall_scores_chart.png')
            plt.savefig(overall_chart_path)
            plt.close()
            visualization_paths.append(overall_chart_path)
            logger.info(f"📊 Generated overall scores chart: {os.path.basename(overall_chart_path)}")
        except Exception as e:
            logger.error(f"❌ Error generating overall score chart: {e}")
        
        # 2. Create category scores radar chart
        try:
            # Prepare data for radar chart
            categories = ['Research\nBasics', 'Methodology', 'Evaluation', 'LLM\nImplementation', 'Critical\nAnalysis']
            
            # Select top 5 papers for clarity
            top_papers = papers_data[:5]
            
            # Set up the radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of categories
            N = len(categories)
            
            # Angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Set the angle labels
            plt.xticks(angles[:-1], categories)
            
            # Draw the lines and fill the areas
            for i, paper in enumerate(top_papers):
                values = [
                    paper['research_basics'],
                    paper['methodology'],
                    paper['evaluation'],
                    paper['llm_implementation'],
                    paper['critical_analysis']
                ]
                values += values[:1]  # Close the loop
                
                # Plot the values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=paper['title'])
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Set y-axis limits
            plt.ylim(0, 100)
            
            # Add title
            plt.title('Category Scores Comparison (Top 5 Papers)')
            
            # Save figure
            radar_chart_path = os.path.join(output_dir, 'category_scores_radar_chart.png')
            plt.savefig(radar_chart_path)
            plt.close()
            visualization_paths.append(radar_chart_path)
            logger.info(f"📊 Generated category scores radar chart: {os.path.basename(radar_chart_path)}")
        except Exception as e:
            logger.error(f"❌ Error generating radar chart: {e}")
        
        # 3. Create average category scores bar chart
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate average scores for each category
            avg_scores = {
                'Research Basics': sum(p['research_basics'] for p in papers_data) / len(papers_data),
                'Methodology': sum(p['methodology'] for p in papers_data) / len(papers_data),
                'Evaluation': sum(p['evaluation'] for p in papers_data) / len(papers_data),
                'LLM Implementation': sum(p['llm_implementation'] for p in papers_data) / len(papers_data),
                'Critical Analysis': sum(p['critical_analysis'] for p in papers_data) / len(papers_data)
            }
            
            categories = list(avg_scores.keys())
            scores = list(avg_scores.values())
            
            # Define color gradient based on scores
            colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
            score_colors = [colors[min(int(score / 20), 4)] for score in scores]
            
            # Create bar chart
            plt.bar(categories, scores, color=score_colors)
            plt.xlabel('Category')
            plt.ylabel('Average Score (%)')
            plt.title('Average Scores by Category')
            
            # Add value labels above each bar
            for i, score in enumerate(scores):
                plt.text(i, score + 1, f"{score:.1f}%", ha='center')
            
            plt.ylim(0, 105)  # Leave space for value labels
            plt.tight_layout()
            
            # Save figure
            category_chart_path = os.path.join(output_dir, 'average_category_scores_chart.png')
            plt.savefig(category_chart_path)
            plt.close()
            visualization_paths.append(category_chart_path)
            logger.info(f"📊 Generated average category scores chart: {os.path.basename(category_chart_path)}")
        except Exception as e:
            logger.error(f"❌ Error generating category scores chart: {e}")
        
        return visualization_paths
    
    def create_quality_report(self, assessment_dir):
        """
        Create a comprehensive quality report from all assessments in a directory.
        
        Args:
            assessment_dir (str): Directory containing assessment JSON files
            
        Returns:
            dict: Quality report summary
        """
        logger.info(f"📊 Creating Quality Report for {assessment_dir}")
        
        # Find all assessment files
        assessment_files = list(Path(assessment_dir).glob("quality_assessment_*.json"))
        logger.info(f"📊 Found {len(assessment_files)} assessment files")
        
        if not assessment_files:
            logger.error(f"❌ No assessment files found in {assessment_dir}")
            return None
            
        # Load all assessments
        assessments = []
        for file_path in assessment_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    assessment = json.load(f)
                    # Calculate score and add it to the assessment
                    assessment['score'] = self.calculate_paper_score(assessment)
                    # Save the updated assessment with scores
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(assessment, f, ensure_ascii=False, indent=2)
                    assessments.append(assessment)
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        logger.info(f"✅ Loaded {len(assessments)} assessments")
        
        # Calculate metrics
        metrics = {
            'total_papers': len(assessments),
            'criteria_scores': {
                'QQ1_Research_Objective': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ2_Method_Description': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ3_Research_Context': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ4_Evaluation_Design': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ5_Metrics': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ6_Baseline_Comparison': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ7_Results_Presentation': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ8_LLM_Integration': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ9_Research_Contribution': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ10_Limitations_Discussion': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ11_Future_Work': {'Yes': 0, 'Partial': 0, 'No': 0},
                'QQ12_Input_Output_Definition': {'Yes': 0, 'Partial': 0, 'No': 0}
            },
            'programming_languages': {},
            'application_types': {},
            'evaluation_types': {},
            'llm_models': {},
            'llm_roles': {},
            'contribution_types': {},
            'limitations_categories': {
                'dataset_limitations': 0,
                'methodology_limitations': 0,
                'llm_specific_limitations': 0,
                'generalizability_limitations': 0,
                'evaluation_limitations': 0
            },
            'future_work_categories': {},
            'scoring_stats': {
                'overall': {
                    'min': 100.0,
                    'max': 0.0,
                    'avg': 0.0,
                    'median': 0.0,
                    'by_letter_grade': {},
                    'total': 0.0
                },
                'categories': {
                    'research_basics': {'min': 100.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0},
                    'methodology': {'min': 100.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0},
                    'evaluation': {'min': 100.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0},
                    'llm_implementation': {'min': 100.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0},
                    'critical_analysis': {'min': 100.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0}
                }
            }
        }
        
        # Process each assessment
        for assessment in assessments:
            qa = assessment.get('assessment', {}).get('quality_assessment', {})
            
            # Count scores for each criterion
            for criterion in metrics['criteria_scores']:
                score = qa.get(criterion, {}).get('score')
                if score in ['Yes', 'Partial', 'No']:
                    metrics['criteria_scores'][criterion][score] += 1
            
            # Count programming languages
            lang = qa.get('QQ3_Research_Context', {}).get('programming_language')
            if lang and lang != 'Not specified':
                metrics['programming_languages'][lang] = metrics['programming_languages'].get(lang, 0) + 1
            
            # Count application types
            app_type = qa.get('QQ3_Research_Context', {}).get('application_type')
            if app_type and app_type != 'Not specified':
                metrics['application_types'][app_type] = metrics['application_types'].get(app_type, 0) + 1
            
            # Count evaluation types
            eval_type = qa.get('QQ4_Evaluation_Design', {}).get('evaluation_type')
            if eval_type and eval_type != 'Not specified':
                metrics['evaluation_types'][eval_type] = metrics['evaluation_types'].get(eval_type, 0) + 1
                
            # Count LLM models used
            llm_models = qa.get('QQ8_LLM_Integration', {}).get('llm_models_used', [])
            if isinstance(llm_models, list):
                for model in llm_models:
                    if model:
                        metrics['llm_models'][model] = metrics['llm_models'].get(model, 0) + 1
            
            # Count LLM roles
            llm_role = qa.get('QQ8_LLM_Integration', {}).get('llm_role')
            if llm_role and llm_role != 'Not specified':
                metrics['llm_roles'][llm_role] = metrics['llm_roles'].get(llm_role, 0) + 1
            
            # Count contribution types
            contribution_types = qa.get('QQ9_Research_Contribution', {}).get('contribution_types', [])
            if isinstance(contribution_types, list):
                for contrib_type in contribution_types:
                    if contrib_type:
                        metrics['contribution_types'][contrib_type] = metrics['contribution_types'].get(contrib_type, 0) + 1
            
            # Count limitations categories
            limitations_categories = qa.get('QQ10_Limitations_Discussion', {}).get('limitations_categories', {})
            if isinstance(limitations_categories, dict):
                for cat_key, cat_values in limitations_categories.items():
                    if cat_key in metrics['limitations_categories'] and isinstance(cat_values, list) and cat_values:
                        metrics['limitations_categories'][cat_key] += len(cat_values)
            
            # Count future work categories
            future_work_cats = qa.get('QQ11_Future_Work', {}).get('future_work_categories', [])
            if isinstance(future_work_cats, list):
                for fw_cat in future_work_cats:
                    if fw_cat:
                        metrics['future_work_categories'][fw_cat] = metrics['future_work_categories'].get(fw_cat, 0) + 1
        
        # Calculate percentages
        for criterion in metrics['criteria_scores']:
            total = sum(metrics['criteria_scores'][criterion].values())
            if total > 0:
                for score in metrics['criteria_scores'][criterion]:
                    metrics['criteria_scores'][criterion][f'{score}_percent'] = round(
                        metrics['criteria_scores'][criterion][score] / total * 100, 1
                    )
        
        # Gather scoring stats
        overall_scores = []
        category_scores = {category: [] for category in metrics['scoring_stats']['categories']}
        
        for assessment in assessments:
            paper_scores = assessment.get('score', {})
            if not paper_scores:
                continue
                
            # Overall scores
            overall_score = paper_scores.get('percentage_score', 0.0)
            overall_scores.append(overall_score)
            
            # Track by letter grade
            letter_grade = paper_scores.get('letter_grade', 'N/A')
            if letter_grade not in metrics['scoring_stats']['overall']['by_letter_grade']:
                metrics['scoring_stats']['overall']['by_letter_grade'][letter_grade] = 0
            metrics['scoring_stats']['overall']['by_letter_grade'][letter_grade] += 1
            
            # Category scores
            for category, scores in paper_scores.get('category_scores', {}).items():
                if category in category_scores:
                    category_scores[category].append(scores.get('percentage', 0.0))
        
        # Calculate overall score statistics
        if overall_scores:
            metrics['scoring_stats']['overall']['min'] = min(overall_scores)
            metrics['scoring_stats']['overall']['max'] = max(overall_scores)
            metrics['scoring_stats']['overall']['avg'] = sum(overall_scores) / len(overall_scores)
            metrics['scoring_stats']['overall']['total'] = sum(overall_scores)
            sorted_scores = sorted(overall_scores)
            middle = len(sorted_scores) // 2
            metrics['scoring_stats']['overall']['median'] = (
                sorted_scores[middle] if len(sorted_scores) % 2 != 0
                else (sorted_scores[middle - 1] + sorted_scores[middle]) / 2
            )
            
        # Calculate category score statistics
        for category, scores in category_scores.items():
            if scores:
                metrics['scoring_stats']['categories'][category]['min'] = min(scores)
                metrics['scoring_stats']['categories'][category]['max'] = max(scores)
                metrics['scoring_stats']['categories'][category]['avg'] = sum(scores) / len(scores)
                metrics['scoring_stats']['categories'][category]['total'] = sum(scores)
        
        # Create output CSV with scores
        try:
            summary_rows = []
            for assessment in assessments:
                qa = assessment.get('assessment', {}).get('quality_assessment', {})
                paper_metadata = assessment.get('paper_metadata', {})
                
                # Function to join list fields with commas
                def join_list(lst):
                    if isinstance(lst, list):
                        return ", ".join(str(x) for x in lst if x)
                    return ""
                    
                # Extract LLM-specific fields
                llm_role = qa.get('QQ8_LLM_Integration', {}).get('llm_role', 'N/A')
                llm_models = join_list(qa.get('QQ8_LLM_Integration', {}).get('llm_models_used', []))
                input_structure = qa.get('QQ8_LLM_Integration', {}).get('input_structure', 'N/A')
                output_structure = qa.get('QQ8_LLM_Integration', {}).get('output_structure', 'N/A')
                
                # Extract contribution fields
                contribution_types = join_list(qa.get('QQ9_Research_Contribution', {}).get('contribution_types', []))
                novelty_level = qa.get('QQ9_Research_Contribution', {}).get('novelty_level', 'N/A')
                
                # Extract limitations fields
                limitations_section = qa.get('QQ10_Limitations_Discussion', {}).get('limitations_section', 'N/A')
                limitations_depth = qa.get('QQ10_Limitations_Discussion', {}).get('limitations_analysis_depth', 'N/A')
                
                # Extract future work fields
                future_work_section = qa.get('QQ11_Future_Work', {}).get('future_work_section', 'N/A')
                research_directions = join_list(qa.get('QQ11_Future_Work', {}).get('research_directions', []))
                
                # Extract input/output fields
                input_clarity = qa.get('QQ12_Input_Output_Definition', {}).get('input_definition_clarity', 'N/A')
                output_clarity = qa.get('QQ12_Input_Output_Definition', {}).get('output_definition_clarity', 'N/A')
                
                # Get scores if available
                paper_scores = assessment.get('score', {})
                overall_score = paper_scores.get('percentage_score', 0.0)
                letter_grade = paper_scores.get('letter_grade', 'N/A')
                
                # Get category scores
                category_scores = paper_scores.get('category_scores', {})
                research_basics_score = category_scores.get('research_basics', {}).get('percentage', 0.0)
                methodology_score = category_scores.get('methodology', {}).get('percentage', 0.0)
                evaluation_score = category_scores.get('evaluation', {}).get('percentage', 0.0)
                llm_implementation_score = category_scores.get('llm_implementation', {}).get('percentage', 0.0)
                critical_analysis_score = category_scores.get('critical_analysis', {}).get('percentage', 0.0)
                
                row = {
                    'title': paper_metadata.get('title', 'Unknown'),
                    'pdf_path': assessment.get('pdf_path', 'Unknown'),
                    
                    # Overall scores
                    'Overall_Score': round(overall_score, 1),
                    'Letter_Grade': letter_grade,
                    
                    # Category scores
                    'Research_Basics_Score': round(research_basics_score, 1),
                    'Methodology_Score': round(methodology_score, 1),
                    'Evaluation_Score': round(evaluation_score, 1),
                    'LLM_Implementation_Score': round(llm_implementation_score, 1),
                    'Critical_Analysis_Score': round(critical_analysis_score, 1),
                    
                    # Individual criterion scores
                    'QQ1_Score': qa.get('QQ1_Research_Objective', {}).get('score', 'N/A'),
                    'QQ2_Score': qa.get('QQ2_Method_Description', {}).get('score', 'N/A'),
                    'QQ3_Score': qa.get('QQ3_Research_Context', {}).get('score', 'N/A'),
                    'QQ4_Score': qa.get('QQ4_Evaluation_Design', {}).get('score', 'N/A'),
                    'QQ5_Score': qa.get('QQ5_Metrics', {}).get('score', 'N/A'),
                    'QQ6_Score': qa.get('QQ6_Baseline_Comparison', {}).get('score', 'N/A'),
                    'QQ7_Score': qa.get('QQ7_Results_Presentation', {}).get('score', 'N/A'),
                    'QQ8_Score': qa.get('QQ8_LLM_Integration', {}).get('score', 'N/A'),
                    'QQ9_Score': qa.get('QQ9_Research_Contribution', {}).get('score', 'N/A'),
                    'QQ10_Score': qa.get('QQ10_Limitations_Discussion', {}).get('score', 'N/A'),
                    'QQ11_Score': qa.get('QQ11_Future_Work', {}).get('score', 'N/A'),
                    'QQ12_Score': qa.get('QQ12_Input_Output_Definition', {}).get('score', 'N/A'),
                    
                    # Context fields
                    'Programming_Language': qa.get('QQ3_Research_Context', {}).get('programming_language', 'N/A'),
                    'Application_Type': qa.get('QQ3_Research_Context', {}).get('application_type', 'N/A'),
                    'Evaluation_Type': qa.get('QQ4_Evaluation_Design', {}).get('evaluation_type', 'N/A'),
                    
                    # LLM integration fields
                    'LLM_Role': llm_role,
                    'LLM_Models': llm_models,
                    'Input_Structure': input_structure,
                    'Output_Structure': output_structure,
                    
                    # Contribution fields
                    'Contribution_Types': contribution_types,
                    'Novelty_Level': novelty_level,
                    
                    # Limitations fields
                    'Limitations_Section': limitations_section,
                    'Limitations_Analysis_Depth': limitations_depth,
                    
                    # Future work fields
                    'Future_Work_Section': future_work_section,
                    'Research_Directions': research_directions,
                    
                    # Input/output fields
                    'Input_Definition_Clarity': input_clarity,
                    'Output_Definition_Clarity': output_clarity
                }
                summary_rows.append(row)
            
            # Create DataFrame and save to CSV
            summary_df = pd.DataFrame(summary_rows)
            csv_path = os.path.join(assessment_dir, "quality_assessment_report.csv")
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"📊 Saved quality report to {csv_path}")
            
            # Save metrics JSON
            metrics_path = os.path.join(assessment_dir, "quality_metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 Saved quality metrics to {metrics_path}")
            
            # Generate visualizations
            visualization_paths = self.generate_score_visualization(assessments, assessment_dir)
            
            return {
                'status': 'success',
                'total_papers': len(assessments),
                'metrics': metrics,
                'report_csv': csv_path,
                'metrics_json': metrics_path,
                'visualizations': visualization_paths
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating quality report: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'total_papers': len(assessments)
            }


# Convenience function for easy access
def assess_papers_from_2023(api_keys, output_dir=None, max_papers=None):
    """
    Convenience function to assess all papers from 2023.
    
    Args:
        api_keys (list): List of API keys for Gemini API
        output_dir (str, optional): Output directory for assessment results
        max_papers (int, optional): Maximum number of papers to assess
        
    Returns:
        dict: Summary of assessment results
    """
    assessor = QualityAssessment(api_keys=api_keys)
    return assessor.assess_2023_papers(
        output_dir=output_dir,
        max_papers=max_papers
    )

def assess_recent_papers(api_keys, years=None, output_dir='filtered_papers/quality_assessments', max_papers=None):
    """
    Convenience function to assess papers from specified years or from 2023 to current year.
    
    Args:
        api_keys (list): List of API keys for Gemini API
        years (list, optional): List of years to assess papers from. If None, uses 2023 to current year.
        output_dir (str): Output directory for assessment results
        max_papers (int, optional): Maximum number of papers to assess per year
        
    Returns:
        dict: Summary of assessment results
    """
    import datetime
    from pathlib import Path
    
    # Create QualityAssessment instance
    assessor = QualityAssessment(api_keys=api_keys)
    
    # Determine years to process
    if years is None:
        current_year = datetime.datetime.now().year
        years = list(range(2023, current_year + 1))
    
    logger.info(f"🔍 Assessing papers from years: {years}")
    
    # Track overall statistics
    total_papers = 0
    successful_assessments = 0
    failed_assessments = 0
    summary_files = []
    
    base_dir = Path('filtered_papers/downloaded_papers')
    
    # Process each year
    for year in years:
        year_dir = base_dir / f"{year}_papers"
        
        # Check if directory exists
        if not year_dir.exists():
            logger.warning(f"⚠️ Directory for {year} papers not found: {year_dir}")
            continue
        
        logger.info(f"📚 Processing papers from {year}")
        
        # Create year-specific output directory
        year_output_dir = Path(output_dir) / f"{year}_assessments"
        year_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Assess papers for this year
        year_summary = assessor.batch_assess_papers(
            pdf_dir=str(year_dir),
            output_dir=str(year_output_dir),
            max_papers=max_papers
        )
        
        if year_summary:
            total_papers += year_summary.get('total_papers', 0)
            successful_assessments += year_summary.get('successful_assessments', 0)
            failed_assessments += year_summary.get('failed_assessments', 0)
            
            if year_summary.get('summary_csv'):
                summary_files.append(year_summary['summary_csv'])
    
    # Create overall summary
    overall_summary = {
        'total_papers': total_papers,
        'successful_assessments': successful_assessments,
        'failed_assessments': failed_assessments,
        'summary_files': summary_files,
        'years_processed': years
    }
    
    logger.info(f"📊 Overall Assessment Summary:")
    logger.info(f"   📄 Total papers: {total_papers}")
    logger.info(f"   ✅ Successful assessments: {successful_assessments}")
    logger.info(f"   ❌ Failed assessments: {failed_assessments}")
    logger.info(f"   📅 Years processed: {years}")
    
    return overall_summary


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Quality Assessment...")
    
    # Example API keys - replace with your own
    API_KEYS = ["AIzaSyANNoXbbOQ4C_q01LYE8_WvUIPU79-80Tc","AIzaSyCsnZT98ULdpJEKGsry5Fv-wnedcRvzV2Y"]
    
    # Initialize quality assessment with API keys
    assessor = QualityAssessment(api_keys=API_KEYS)
    
    # Example: Assess a single paper
    # result = assessor.assess_paper('path/to/your/paper.pdf')
    
    # Example: Batch assess papers from 2023
    # summary = assessor.assess_2023_papers(max_papers=5)
    
    print("✅ Module loaded successfully")
    print("💡 Please configure API keys and call assessment methods directly") 