import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

# Initialize Flask app
app = Flask(__name__) 

# Enable CORS for all routes
CORS(app)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the CTransformers model globally only once during app startup
try:
    llm = CTransformers(
        model=r"D:/ps project/backend/codellama-7b-instruct.Q4_K_M.gguf",
        model_type='llama',
        config={'max_new_tokens': 500, 'temperature': 0.01}
    )
    logger.info("CodeLLaMA model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading CodeLLaMA model: {str(e)}")

# Define function to get response from CodeLLaMA model
def get_llama_response(input_text, timeComplexity, language, temperature=0.01):
    try:
        logger.info("Generating code using CodeLLaMA model...")

        # Define the prompt template
        template = (
            "Generate code for description:\n\n"
            "Description: '{input_text}'\n\n"
            "Time Complexity: {timeComplexity}\n\n"
            "in Programming Language: {language}\n\n"
        )

        # Create the PromptTemplate
        prompt = PromptTemplate(
            input_variables=["input_text", "timeComplexity", "language"],
            template=template
        )

        # Format the prompt
        formatted_prompt = prompt.format(
            input_text=input_text,
            timeComplexity=timeComplexity,
            language=language
        )

        logger.debug(f"Formatted Prompt: {formatted_prompt}")

        # Generate the response using the model
        response = llm.invoke(formatted_prompt)
        
        if not response: 
            raise ValueError("Model returned an empty response")

        # Return the generated response
        return response.strip()  # Remove any unnecessary whitespace

    except Exception as e:
        logger.error(f"Error interacting with CodeLLaMA model: {str(e)}")
        raise e

# Define route to generate code
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        logger.info("POST request received at /generate")

        # Extract data from request
        data = request.json
        input_text = data.get('input_text')
        timeComplexity = data.get('timeComplexity')
        language = data.get('language')

        # Validate inputs
        if not all([input_text, timeComplexity, language]):
            raise ValueError("Missing required fields in request: input_text, timeComplexity, language")

        logger.debug(f"Input Text: {input_text}, Time Complexity: {timeComplexity}, Language: {language}")

        # Get the response from the CodeLLaMA model
        response = get_llama_response(input_text, timeComplexity, language)

        logger.info("Successfully generated code.")
        return jsonify({'response': response}), 200

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400  # Bad Request

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500  # Internal Server Error

# Run the Flask app
if __name__ == '__main__':
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)
