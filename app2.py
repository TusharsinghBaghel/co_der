from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
import os
import json
from dotenv import load_dotenv
import re
from crewai_tools import SerperDevTool

# Initialize SerperDevTool for link search
linkstool = SerperDevTool(n_results=2)

# Load environment variables
load_dotenv()

# Validate and set Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

SERPERKEY = os.getenv("SERPER_API_KEY")
if not SERPERKEY:
    raise RuntimeError("SERPER_API_KEY not set in environment variables.")
os.environ["SERPER_API_KEY"] = SERPERKEY

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize LLM configuration
llama3_70b = LLM(
    model="groq/llama3-70b-8192",
    temperature=0.3,
    max_completion_tokens=2000,
    top_p=1,
    stream=False,
)

llama3_instant = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.3,
    max_completion_tokens=2000,
    top_p=1,
    stream=False,
)

# Define Master Agent
master_agent = Agent(
    knowledge_sources=[],
    role="Orchestrator",
    goal="Analyze queries and create execution plans",
    backstory="Expert workflow designer",
    verbose=True,
    llm=llama3_instant,
)

# Specialized Agents
concept_agent = Agent(
    role="Concept Teacher",
    goal="Explain programming concepts using the tools to search for blogs and articles",
    backstory="Expert technical educator",
    verbose=True,
    llm=llama3_instant,
)

code_agent = Agent(
    role="Code Tutor",
    goal="Provide code examples and templates using docs search and user specific docs like tutoring",
    backstory="Senior developer mentor",
    verbose=True,
    llm=llama3_instant,
)

debug_agent = Agent(
    role="Debug Guide",
    goal="Troubleshoot code issues",
    backstory="Experienced debug specialist",
    verbose=True,
    llm=llama3_instant,
)

progress_agent = Agent(
    role="Progress Tracker",
    goal="Track learning progress",
    backstory="Learning analytics expert",
    verbose=True,
    llm=llama3_instant,
    memoryview=True,
)

def extract_json(text):
    """
    Extract and fix JSON from a possibly messy LLM response.
    Handles unescaped characters and markdown formatting.
    """
    try:
        # Remove markdown backticks/code blocks if present
        cleaned = re.sub(r"```(?:json|python)?\n?", "", text).strip("`")

        # Match the first JSON object
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start == -1 or end == -1 or start > end:
            raise ValueError("No JSON object found.")
        
        json_str = cleaned[start:end + 1]

        # Fix unescaped control characters (e.g., \n, \t)
        fixed_str = json_str.encode("utf-8", "ignore").decode("unicode_escape")

        # Load the JSON safely
        return json.loads(fixed_str)

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON extraction failed: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

def run_agent(agent, prompt):
    """Helper function to execute agent tasks"""
    task = Task(
        description=prompt,
        expected_output="Well-structured response in markdown",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    return crew.kickoff()

@app.route('/callMaster', methods=['POST'])
def master_endpoint():
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        prompt = f"""
        Analyze this developer query and generate a step-by-step execution plan with agent roles and prompts.

        The 4 agents are concept, code, debug, and progress.
        concept: Explain programming concepts and algorithms.
        code: Provide code examples and implementations.
        debug: Help troubleshoot code issues and errors.
        progress: Call the progress agent to track learning progress. give the list of concepts and skills the user will learn through this particular prompt of agent actions of concept builder, debug and code tutor.
        Call each agent only once and only if needed. Call the progress agent at the end of the prompt everytime.
        No need to call debug agent if there are no errors in the code, if the user is just asking for an explanation or code example.
        No need to call concept agent if the user is just asking for a code example or debugging help. Although if the code or error debug involves a concept, you can call the concept agent to explain it.
        No need to call code agent if the user is just asking for an explanation or debugging help.

        The prompt generation for each agent should be as follows:
        concept: Ask the concept agent to explain the concept or algorithm in detail. No need of code
        code: Ask the code agent to provide a code example or snippets, boiler plates of the functions, that enables him to write his own code easily
        debug: Ask the debug agent to help troubleshoot the code issues and errors using stack overflow or other debugging communities and resources.
        progress: In the prompt send the list of concepts and skills the user will learn through this particular prompt of agent actions of concept builder, debug and code tutor.

        QUERY: {query}

        IMPORTANT: 
        Your final response MUST be ONLY the valid JSON.
        Do NOT include markdown, comments, or explanation.
        Just return a JSON object like this:

        {{
        "plan": [
            {{
            "agent": "concept",
            "prompt": "Explain the quicksort algorithm..."
            }},
            {{
            "agent": "code",
            "prompt": "Write the implementation in Python..."
            }}
        ]
        }}
        """

        result = run_agent(master_agent, prompt)
        result = extract_json(str(result))
        print(f"Master Agent Result: {result}")
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/agent/concept', methods=['POST'])
def concept_agent_endpoint():
    data = request.json
    initial_prompt = data.get('prompt', '')
    
    prompt = f"""
    You are a concept tutor agent. Follow these rules STRICTLY:
    1. Use \\\\n for new lines in explanations (NOT actual newlines)
    2. Keep explanations as SINGLE-LINE strings
    3. Escape all double quotes with \\"
    4. Return ONLY a valid JSON, No headings,No comments, NO explanations, No thoughts.

    Your task is to:
    1. Explain: {initial_prompt}
    2. Generate a realistic search query developers would use for this query to search for blogs and good understanding of the concept.
    3. Provide a Json object with two keys:

    Required JSON format:
    {{
        "search_text": "concept keywords to search for concepts contents and blogs",
        "explanation": "Concept overview\\\\nKey points:\\\\n1. Point one\\\\n2. Point two"
    }}

    Example valid output:
    {{
        "search_text": "dynamic programming memoization and tabulation",
        "explanation": "Dynamic programming optimizes recursive solutions.\\\\nKey aspects:\\\\n1. Memoization: Top-down approach with cache\\\\n2. Tabulation: Bottom-up table building\\\\n3. Optimal substructure requirement"
    }}
    """

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        result = run_agent(concept_agent, prompt)
        result = extract_json(str(result))
        search_text = result.get('search_text', '')
        links = linkstool.run(search_query=search_text)
        result['links'] = links
        print(f"Concept Agent Result: {result}")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent/code', methods=['POST'])
def code_agent_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    formatted_prompt = f"""
    You are a code assistant AI agent.
    
    Your job is to generate high-quality, minimal, code hints templates for the user query.
    Additionally, generate a realistic Google search text that a skilled senior developer would have used to find the **exact documentation or examples** needed to solve this query.
    IMPORTANT:
    1. Use \\\\n for new lines in Code (NOT actual newlines)
    2. Keep explanations as SINGLE-LINE strings
    3. Escape all double quotes with \\"
    4. Return ONLY a valid JSON, No headings,No comments, NO explanations, No thoughts.

    Your final response MUST be a JSON object with exactly two keys:
    1. "search_text": a concise but specific search phrase someone would use to find the right doc/resource.
    2. "code": the code snippet solving the query.

    Avoid markdown formatting or any comments. Only return a JSON object.

    QUERY: {prompt}

    EXAMPLE OUTPUT:
    {{
        "search_text": "python requests post json data with headers",
        "code": "import requests\\\\nheaders = {{'Content-Type': 'application/json'}}\\\\nresponse = requests.post('https://example.com', json={{'key': 'value'}}, headers=headers)"
    }}
    """
    if not formatted_prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        result = run_agent(code_agent, formatted_prompt)
        result = extract_json(str(result))
        #convert string to json
        
        search_text = result.get('search_text', '')
        links = linkstool.run(search_query=search_text)
        result['links'] = links
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent/debug', methods=['POST'])
def debug_agent_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    formatted_prompt = f"""
    You are a debug assistant agent.

    Your task is to help the user understand and debug the following error or issue. 
    Along with the explanation, generate a realistic and effective Stack Overflow / Google search query that a skilled senior developer might use to search for a solution.

    Your final output MUST be a valid JSON object with exactly two keys:
    1. "search_text": a concise and accurate query a developer would use to find similar issues or solutions online.
    2. "explanation": the reasoning behind the error and how to fix or debug it.
    3. Use \\\\n for new lines in explanations (NOT actual newlines)


    Do NOT include markdown, comments, or any text outside the JSON block.

    QUERY: {prompt}

    EXAMPLE OUTPUT:
    {{
        "search_text": "TypeError cannot read property of undefined javascript async await",
        "explanation": "This error typically occurs when trying to access a property of an undefined object. This can happen if a value hasn’t loaded yet due to asynchronous behavior. To fix this, ensure the object is properly initialized or check for undefined before accessing its properties."
    }}
    """
    if not formatted_prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        result = run_agent(debug_agent, formatted_prompt)
        result = extract_json(str(result))
        search_text = result.get('search_text', '')
        links = linkstool.run(search_query=search_text)
        result['links'] = links
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent/progress', methods=['POST'])
def progress_agent_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    formatted_prompt = f"""
    You are a progress tracking agent. You take as input the list of concepts and skills the user will learn through this particular prompt of agent actions of concept builder, debug and code tutor.
    And return the updated list of concepts from the old memory and the new concepts learned through this prompt.
    Your final output MUST be a valid JSON object with exactly two keys:
    1. "Current Concepts": a list of concepts the user has learned so far.
    2. "Recommendations": a list of new concepts the user should learn next based on the current concepts and the prompt.
    3. Use \\\\n for new lines inside the value/list (NOT actual newlines)
    Do NOT include markdown, comments, or any text outside the JSON block.

    New learnings: {prompt}

    EXAMPLE OUTPUT:
    {{
        "Current Concepts": ["skill/concept 1", "skill/concept 2", "skill/concept 3"],
        "Recommendations": ["recommendation1", "recommendation2", "recommendation3"]
    }}

    """
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        result = run_agent(progress_agent, formatted_prompt)
        result = extract_json(str(result))
        return result
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent/search', methods=['POST'])
def serper_agent_endpoint():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        result = linkstool.run(search_query=prompt)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5328, debug=True)
