import time
from google import genai
import PyPDF2
import os
import json
import re

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
        
        # JSON structure template for paper analysis
        self.json_structure = """
    {
        "paper_identification": {
            "title": "Extract the title of the paper exactly as it appears.",
            "authors": "List all authors in the order they are presented, including their affiliations if provided.",
            "publication_venue_year": "Specify the publication venue (e.g., conference, journal) and the year of publication."
        },
        "abstract_analysis": {
            "problem_statement": {
                "analysis": "Identify and describe the core problem in Text-to-SQL that the paper aims to solve, as explicitly stated in the abstract. Focus on the specific challenge or gap the authors highlight, such as limitations in natural language understanding, schema complexity, or query accuracy.",
                "evidence": "Provide a direct quote or a precise paraphrase from the abstract that clearly defines the problem."
            },
            "proposed_methodology_summary": {
                "analysis": "Summarize the core methodology or system proposed by the authors to address the identified problem. Include the key components or techniques mentioned in the abstract, such as neural network architectures, prompt engineering, or retrieval mechanisms.",
                "evidence": "Quote or specifically paraphrase the part of the abstract that outlines the proposed approach."
            },
            "key_achievements_summary": {
                "analysis": "Summarize the main achievements, key results, or the claimed 'goodness' of their approach as highlighted in the abstract. This may include performance improvements (e.g., accuracy gains), novel capabilities (e.g., handling complex queries), or other significant outcomes (e.g., scalability).",
                "evidence": "Quote or specifically paraphrase the abstract's statements about the results or contributions."
            }
        },
        "introduction_analysis": {
            "background": {
                "analysis": "Describe the general background context provided in the introduction, including any historical developments (e.g., evolution of Text-to-SQL systems), current challenges (e.g., schema ambiguity, multi-table queries), or the broader problem area (e.g., natural language interfaces to databases) that sets the stage for the specific research problem. Highlight how this context relates to the field of Text-to-SQL.",
                "evidence": "Quote or specifically paraphrase from the introduction to support the background description."
            },
            "literature_review": {
                "analysis": "Conduct an in-depth analysis of the literature review, which may be integrated into the introduction or presented separately. Focus on the following aspects:\\n- Identify and categorize the existing Text-to-SQL approaches discussed (e.g., rule-based systems, statistical models, neural networks, hybrid methods).\\n- For each major approach or key study mentioned, summarize:\\n    - The core methodology or technique used (e.g., parsing techniques, encoder-decoder models).\\n    - The main findings or contributions (e.g., accuracy on specific benchmarks).\\n    - Any reported limitations or challenges (e.g., inability to generalize across datasets).\\n- Describe how the current paper builds upon, extends, or differs from these existing approaches (e.g., by introducing a novel component or addressing a specific limitation).\\n- Highlight any gaps or unresolved issues in the literature that the paper aims to address (e.g., lack of robustness in real-world scenarios).\\n- If applicable, note any comparative analyses or discussions of the strengths and weaknesses of different approaches provided by the authors.\\n- Observe any trends or shifts in the field as reflected in the literature discussed (e.g., the move from rule-based to neural approaches).\\n- If the literature review is extensive, prioritize summarizing the most relevant or influential works that directly relate to the paper's contributions.",
                "evidence": "Provide quotes or specific paraphrases from the relevant section(s) to support each point of analysis."
            },
            "extended_detail_on_context": {
                "analysis": "Identify and describe any specific aspects of the background or prior work that the paper extends or details further. This could include deeper explanations of certain techniques (e.g., how schema linking is performed), additional context on challenges (e.g., handling nested queries), or clarifications on the problem space (e.g., specific database types).",
                "evidence": "Quote or specifically paraphrase from the introduction to illustrate these extensions."
            },
            "problem_definition_detailed": {
                "analysis": "Explain how the paper formally or in-depth defines the Text-to-SQL problem it addresses within the introduction. Include any specific constraints (e.g., single-table vs. multi-table queries), assumptions (e.g., schema availability), or scope limitations (e.g., focus on specific query types) mentioned.",
                "evidence": "Quote or specifically paraphrase the detailed problem definition from the introduction."
            },
            "existing_approaches_overview": {
                "analysis": "Summarize the general categories (e.g., syntax-based, semantics-based) or specific examples (e.g., Seq2SQL, IRNet) of existing Text-to-SQL approaches mentioned in the introduction. Focus on how these approaches are characterized or grouped by the authors.",
                "evidence": "Quote or specifically paraphrase from the introduction to support the overview."
            },
            "limitations_of_existing_approaches": {
                "analysis": "Detail the limitations or shortcomings of existing approaches as pointed out by the authors. Include specific examples (e.g., failure on complex joins) or categories of issues (e.g., poor generalization, high error rates) that motivate the current research.",
                "evidence": "Quote or specifically paraphrase the discussion of limitations from the introduction."
            },
            "claimed_contributions": {
                "analysis": "Identify and describe the specific, novel contributions the paper claims to make to the field of Text-to-SQL. Focus on what sets this work apart from prior research, such as new techniques, improved performance, or addressing previously unsolved problems.",
                "evidence": "Quote or specifically paraphrase the statements of contributions from the introduction."
            },
            "introduction_conclusion_roadmap": {
                "analysis": "Determine if the introduction concludes with a roadmap or outline of the paper's structure. If present, summarize the key sections or flow of the paper as described (e.g., methodology, experiments, discussion).",
                "evidence": "Quote or specifically paraphrase the roadmap or outline from the introduction."
            }
        },
        "methodology_analysis": {
            "ai_techniques_used": {
                "analysis": "Identify and describe any specific AI techniques mentioned in the methodology, such as deep learning architectures (e.g., transformers, RNNs), reinforcement learning, or specific types of neural networks (e.g., BERT-based models). Explain how these techniques are applied in the context of Text-to-SQL (e.g., query generation, schema encoding).",
                "evidence": "Quote or specifically paraphrase from the methodology section to support the description."
            },
            "input_output_specifications": {
                "analysis": "Describe the precise input to the system (e.g., natural language question format, database schema representation such as JSON or SQL DDL) and the expected output (e.g., SQL query syntax, including specific clauses like SELECT, WHERE). Include any preprocessing (e.g., tokenization) or postprocessing steps (e.g., query validation) mentioned.",
                "evidence": "Quote or specifically paraphrase from the methodology section to detail the input and output."
            },
            "algorithm_llm_prompt_details": {
                "analysis": "Provide a detailed explanation of the proposed algorithm or model architecture. If a Large Language Model (LLM) is used, specify which LLM (e.g., GPT-3, LLaMA), any fine-tuning processes (e.g., dataset used, training objectives), and critically, the details of prompt engineering (e.g., few-shot examples, chain-of-thought reasoning, schema linking instructions, specific prompt templates like 'Given this schema: {schema}, generate an SQL query for: {question}').",
                "evidence": "Quote or specifically paraphrase from the methodology section to support the explanation."
            },
            "rag_usage_details": {
                "analysis": "If Retrieval Augmented Generation (RAG) or similar techniques are used for schema or knowledge integration, explain how they are implemented and utilized within the system. Include details on the retrieval mechanism (e.g., vector search over schema elements), how retrieved information is integrated (e.g., into prompts), and how it enhances the Text-to-SQL process (e.g., improving accuracy on unseen schemas).",
                "evidence": "Quote or specifically paraphrase from the methodology section to describe the implementation."
            },
            "workflow_details": {
                "analysis": "Describe the step-by-step workflow or pipeline of how a natural language question is processed and converted to an SQL query by the system. Include stages like question parsing, schema mapping, query generation, and validation. If multiple workflows or variations are presented (e.g., with/without RAG), detail each one.",
                "evidence": "Quote or specifically paraphrase from the methodology section to outline the workflow."
            },
            "control_mechanisms_or_experiments": {
                "analysis": "Identify and describe any control mechanisms (e.g., baseline comparisons), ablation studies (e.g., removing prompt engineering), or variations of the method tested (e.g., different LLMs) to isolate the contributions of different components. Explain the purpose (e.g., to measure impact of schema linking) and findings of these experiments.",
                "evidence": "Quote or specifically paraphrase from the methodology or experimental setup section."
            },
            "complexity_of_method": {
                "analysis": "Discuss any mentions of the complexity of the proposed method, whether computational (e.g., time complexity O(n), memory usage), architectural (e.g., number of parameters, layers), or conceptual (e.g., difficulty in implementation or interpretability). Include trade-offs if mentioned.",
                "evidence": "Quote or specifically paraphrase from the methodology section to support the discussion."
            },
            "presentation_of_methodology": {
                "analysis": "Describe how the methodology is presented in the paper, such as through tables (e.g., parameter settings), figures (e.g., architecture diagrams), lists (e.g., workflow steps), or pseudo-code (e.g., algorithm outline). Briefly explain any key visual or structural aids that help clarify the method.",
                "evidence": "Reference specific figures, tables, or sections, or quote descriptions from the text."
            }
        },
        "results_analysis": {
            "evaluation_criteria_and_metrics": {
                "analysis": "Identify and describe the specific evaluation criteria and metrics used to assess the performance of the Text-to-SQL system (e.g., Execution Accuracy, Exact Set Match Accuracy, F1-score for query components). Specify the standard Text-to-SQL benchmarks (e.g., Spider, WikiSQL, SParC, CoSQL) on which the results are reported, including dataset characteristics (e.g., complexity, size).",
                "evidence": "Quote or specifically paraphrase from the results or experimental setup section."
            },
            "quantitative_results": {
                "analysis": "Present the key quantitative results reported in the paper, including specific scores (e.g., 85% accuracy), improvements over baselines (e.g., +5% over previous SOTA), and any statistical significance (e.g., p-values) mentioned. Include comparisons to other methods if provided.",
                "evidence": "Directly quote or accurately summarize the numerical results and associated benchmarks/metrics."
            }
        },
        "discussion_analysis": {
            "interpretation_of_results_how_good": {
                "analysis": "Explain how the authors interpret their results, including how 'good' they claim their method is compared to state-of-the-art (SOTA) methods or baselines. Discuss any qualitative assessments (e.g., robustness) or implications (e.g., real-world applicability) of the results.",
                "evidence": "Quote or specifically paraphrase from the discussion section to support the interpretation."
            },
            "limitations_of_the_study": {
                "analysis": "Identify and describe the explicitly stated or discernible limitations, weaknesses, or failure cases of the proposed method or the study itself, as discussed by the authors. Include specific examples (e.g., struggles with nested queries) or scenarios (e.g., low performance on certain benchmarks).",
                "evidence": "Quote or specifically paraphrase from the discussion section."
            },
            "explanation_of_performance_why": {
                "analysis": "Determine if the paper offers explanations, hypotheses, or reasons for why their method achieves certain results, both positive (e.g., effective schema encoding) and negative (e.g., prompt sensitivity). For instance, why it performs well on certain types of queries (e.g., simple selects) or datasets (e.g., Spider), or why it fails in others (e.g., multi-table joins).",
                "evidence": "Quote or specifically paraphrase from the discussion section to support the explanations."
            },
            "deeper_analysis_insights": {
                "analysis": "Identify any deeper analysis of the results, such as error analysis (e.g., types of SQL errors like incorrect joins, common failure modes like misinterpreting ambiguity), case studies (e.g., specific query examples), or qualitative insights into the system's behavior (e.g., over-reliance on training patterns). Describe the key findings from this analysis.",
                "evidence": "Quote or specifically paraphrase from the discussion or error analysis section."
            },
            "llm_specific_discussion": {
                "analysis": "If the method is LLM-based, identify and describe any specific discussions around LLM behavior, such as generalization capabilities (e.g., across datasets), understanding of complex schemas (e.g., multi-table relationships), sensitivity to prompts (e.g., rephrasing effects), hallucination issues (e.g., generating invalid SQL), or ethical considerations (e.g., bias in query interpretation).",
                "evidence": "Quote or specifically paraphrase from the discussion section focusing on LLM aspects."
            }
        },
        "conclusion_analysis": {
            "summary_of_achievements": {
                "analysis": "Summarize the main achievements, findings, and takeaways of the research as stated in the paper's conclusion section. Focus on the key points the authors want to leave with the reader, such as novel contributions, performance gains, or practical impacts.",
                "evidence": "Quote or specifically paraphrase from the conclusion section."
            },
            "llm_practical_considerations": {
                "analysis": "If an LLM is central to the method, determine if the conclusion touches upon practical aspects such as accuracy-cost-latency tradeoffs (e.g., inference time vs. performance), stability (e.g., consistency across runs), scalability (e.g., handling large schemas), or security implications (e.g., injection risks) for real-world deployment of their Text-to-SQL system.",
                "evidence": "Quote or specifically paraphrase from the conclusion section regarding LLM practicalities."
            }
        },
        "future_directions_analysis": {
            "proposed_future_work_potential_solutions": {
                "analysis": "Identify and describe the potential avenues for future research, improvements to their method, or solutions to remaining challenges in Text-to-SQL as suggested by the authors. Include any specific directions (e.g., incorporating external knowledge) or open questions (e.g., handling dynamic schemas) mentioned.",
                "evidence": "Quote or specifically paraphrase from the future work or conclusion section."
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
            print("Rate limit reached (15 requests/minute). Waiting 65 seconds...")
            time.sleep(65)
            self.requests_in_minute = 0
            self.minute_start_time = time.time()
        
        # Rotate key after max_requests_per_key requests
        if self.request_count >= self.max_requests_per_key:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.request_count = 0
            print(f"Switched to API key: {self.api_keys[self.current_key_index][:10]}...")
        
        self.request_count += 1
        self.requests_in_minute += 1
        return self.api_keys[self.current_key_index]
    
    def create_prompt_content(self, paper_content):
        """
        Create the prompt content for the Gemini API.
        
        Args:
            paper_content (str): Paper content to analyze
            
        Returns:
            str: Formatted prompt for the Gemini API
        """
        prompt_string = (
        "You are an expert research paper analyst specializing in Natural Language Processing, particularly Text-to-SQL systems. "
        "Your primary task is to perform a **thorough, in-depth, and comprehensive analysis** of the provided Text-to-SQL research paper. "
        "You must generate a detailed report in JSON format. \n\n"
        "**Crucially, for EACH AND EVERY analytical point and for EVERY field within the JSON structure that requires textual interpretation, you MUST provide direct quotes or highly specific paraphrased evidence extracted directly from the paper.** "
        "This evidence is non-negotiable. Clearly cite the section or page number for each piece of evidence, if available. "
        "Ensure that your analysis covers all relevant aspects presented in the paper, leaving no significant detail unaddressed. "
        "Generic statements or analyses without direct textual backing from the paper are unacceptable.\n\n"
        "**When extracting text from the paper for inclusion in the JSON, ensure that:**\n"
        "- All special characters are properly escaped according to JSON standards (e.g., double quotes should be escaped with a backslash).\n"
        "- Any formatting issues in the paper's text are handled to prevent JSON syntax errors (e.g., remove or replace problematic characters).\n"
        "- Text is treated as plain strings, without interpreting any content as JSON syntax.\n"
        "- Use UTF-8 encoding for any non-ASCII characters to ensure compatibility.\n"
        "- Remove or replace any control characters or non-printable characters that could invalidate the JSON.\n"
        "- While handling the text for JSON compatibility, ensure that the meaning and accuracy of the extracted evidence are preserved. Do not alter the text in a way that changes its original intent or content.\n\n"
        "**Additionally, ensure that all analysis is strictly based on the content extracted from the paper. Do not use external knowledge or make inferences that extend beyond what is explicitly stated in the text. Your analysis must remain focused solely on the information provided within the paper.**\n\n"
        "The JSON output must **strictly adhere** to the following structure, mirroring a detailed breakdown of a research paper:\n\n" +
        self.json_structure +
        "\n\nProvide your detailed and evidence-backed analysis for the following paper content:\n\n" +
        paper_content +
        "\n\n**Before finalizing your analysis, review each point to ensure it is directly supported by the paper's content.**\n"
        "**Additionally, verify that the generated JSON is structurally valid and that all text fields are properly formatted and escaped.**\n\n"
        "Your Response:\n"
        )

        return prompt_string
    
    def analyze_content_with_gemini(self, paper_content, api_key=None):
        """
        Analyze paper content using the Gemini API.
        
        Args:
            paper_content (str): Paper content to analyze
            api_key (str, optional): API key to use, if None, get from rotation
            
        Returns:
            dict: Parsed JSON result from Gemini API
        """
        if not api_key:
            api_key = self.get_current_api_key()
            
        client = genai.Client(api_key=api_key)
        prompt_content = self.create_prompt_content(paper_content)
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=prompt_content
            )
            result = response.text.strip()
            print("Gemini API request successful")

            # Extract JSON from markdown code block
            match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    json_data = json.loads(json_str)
                    return json_data  # Return parsed dictionary
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from response: {e}")
                    return None
            else:
                print("No JSON found in the response")
                return None
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text from PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def save_json_output(self, file_path, json_result):
        """
        Save JSON result to file.
        
        Args:
            file_path (str): Path to the original file
            json_result (dict): JSON result to save
            
        Returns:
            str: Path to saved JSON file
        """
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{base_name}.json"
        output_full_path = os.path.join(os.path.dirname(file_path), output_file_name)
        
        # Check if json_result is valid (not None and is a dict)
        if json_result is not None and isinstance(json_result, dict):
            with open(output_full_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_result, json_file, ensure_ascii=False, indent=4)
            print(f"Results saved to {output_full_path}")
            return output_full_path
        else:
            print(f"Cannot save results for {file_path}: json_result is invalid")
            return None
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Path to saved JSON file, or None if processing failed
        """
        print(f"Processing {pdf_path}...")
        
        # Check if JSON already exists
        json_path = os.path.splitext(pdf_path)[0] + '.json'
        if os.path.exists(json_path):
            print(f"JSON file already exists for {pdf_path}, skipping...")
            return json_path
        
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                print(f"No text extracted from {pdf_path}")
                return None
                
            # Analyze content
            api_key = self.get_current_api_key()
            json_result = self.analyze_content_with_gemini(paper_content=text, api_key=api_key)
            
            # Save result
            if json_result is not None:
                return self.save_json_output(file_path=pdf_path, json_result=json_result)
            else:
                print(f"No JSON result for {pdf_path}")
                return None
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_directory(self, directory_path):
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path (str): Path to directory containing PDF files
            
        Returns:
            list: List of paths to saved JSON files
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"{directory_path} is not a valid directory.")
        
        # Ensure directory exists
        os.makedirs(directory_path, exist_ok=True)
        
        # Get all PDF files in directory
        pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process each PDF file
        results = []
        for pdf_file in pdf_files:
            result = self.process_pdf(pdf_file)
            if result:
                results.append(result)
        
        print(f"Processed {len(results)} out of {len(pdf_files)} PDF files")
        return results


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with API keys
    analyzer = ContentAnalyzer(api_keys=["your_api_key_here"])
    
    # Process a directory of PDF files
    directory_path = "arxiv"
    results = analyzer.process_directory(directory_path)
