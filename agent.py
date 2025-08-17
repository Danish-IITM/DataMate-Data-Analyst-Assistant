# agent.py (Final version with robust error handling and correct Matplotlib setup)

import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import traceback
import io
import sys
import base64
import json
import re

def execute_python_code(code: str) -> dict:
    """Executes a complete string of Python code and returns its stdout and stderr."""
    # Use a fresh state for each execution to ensure scripts are self-contained.
    # We will pass required libraries into the execution context.
    execution_globals = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "matplotlib": __import__("matplotlib"),
        "plt": __import__("matplotlib.pyplot"),
        "io": __import__("io"),
        "base64": __import__("base64"),
        "json": __import__("json"),
    }
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    
    try:
        exec(code, execution_globals)
    except Exception:
        tb_string = traceback.format_exc()
        sys.stderr.write(tb_string)
    finally:
        captured_stdout = sys.stdout.getvalue()
        captured_stderr = sys.stderr.getvalue()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        
    return {"stdout": captured_stdout, "stderr": captured_stderr}

class DataAnalystAgent:
    def __init__(self, api_key: str, work_dir: str):
        genai.configure(api_key=api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            safety_settings=safety_settings
        )
        self.work_dir = work_dir
        self.request_options = {"timeout": 240.0}

    def run(self, question: str, files: list) -> str:
        original_dir = os.getcwd()
        os.chdir(self.work_dir)

        file_list_str = ", ".join(files) if files else "No files were uploaded."
        
        # --- DEFINITIVE PROMPT WITH MATPLOTLIB FIX ---
        prompt = f"""You are an expert Python data analyst. Your task is to write a single, self-contained Python script to answer the user's request.

User's files available in the current directory: [{file_list_str}]
User's request:
---
{question}
---

CRITICAL INSTRUCTIONS:
1.  Your output MUST be a single Python script wrapped in ```python ... ```.
2.  **MATPLOTLIB SETUP:** If you need to generate a plot, the first three lines of your script MUST be exactly:
    ```python
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ```
    This prevents server errors.
3.  **PLOT GENERATION:** The script must generate plots, save them to an in-memory buffer (`io.BytesIO`), encode them to a base64 string, and format as a data URI (`data:image/png;base64,...`).
4.  **FINAL OUTPUT:** The **very last line of your script** MUST be a single `print()` statement that outputs a valid JSON array or object containing all the answers. Convert numpy types (like np.int64) to standard Python types (int, float) before creating the JSON.

Example of a final line for a script requesting a number, a string, and a plot:
`print(json.dumps([int(some_numpy_number), "some_string", plot_data_uri]))`
"""
        
        try:
            print("Calling Gemini API to generate Python script...")
            response = self.model.generate_content(
                prompt,
                request_options=self.request_options
            )
            
            if not response.candidates:
                raise ValueError(f"API call failed: The request was likely blocked. Feedback: {response.prompt_feedback}")
            
            code_to_execute = response.text
            match = re.search(r"```(python\s*)?([\s\S]+)```", code_to_execute, re.MULTILINE)
            if match:
                code_to_execute = match.group(2).strip()
            else:
                code_to_execute = code_to_execute.strip()
            
            print(f"--- Agent Generated This Script ---\n{code_to_execute}\n-----------------------------------")
            
            result = execute_python_code(code_to_execute)

            # --- DEFINITIVE FIX FOR ERROR HANDLING ---
            # Only raise an exception if stderr contains a 'Traceback', ignoring warnings.
            if "Traceback" in result["stderr"]:
                raise Exception(f"Error during Python script execution: {result['stderr']}")
            
            final_json_output = result["stdout"].strip()
            
            if not final_json_output:
                # If there's no output but there was a warning, show the warning.
                if result["stderr"]:
                    raise ValueError(f"The generated script did not print any output. Warnings: {result['stderr']}")
                raise ValueError("The generated script did not print any output.")

            json.loads(final_json_output)
            print(f"SUCCESS: Script produced valid JSON output.")
            return final_json_output

        except (google_exceptions.RetryError, google_exceptions.DeadlineExceeded) as e:
            print(f"API Timeout/RetryError: The service is likely overloaded. {e}")
            raise 
        except Exception as e:
            print(f"An unexpected error occurred in agent.run: {e}")
            traceback.print_exc()
            raise
        finally:
            os.chdir(original_dir)