"""
AgenticAI.py
This module implements an orchestrator for an agentic AI sentiment analysis system.
It aggregates predictions from multiple AI models and uses a meta-model to make final predictions.
"""

import pandas as pd
import logging
import time
from GPTmethods import GPTmethods
from GEMINImethods import GEMINImethods
from CLAUDEmethods import CLAUDEmethods
from DEEPSEEKmethods import DEEPSEEKmethods
# rm -rf .venv
# python3 -m venv .venv
# .venv/bin/python --version
# .venv/bin/pip install pandas openai python-dotenv
# /Users/rkonstadinos/PycharmProjects/AgenticRecommender/.venv/bin/python Classes/AgenticAI.py
class AgenticAI:
    def __init__(self, model_name, datasets_path='Datasets/', dataset_file='Dataset.csv'):
        """
        Sets up the orchestrator that coordinates multiple AI agents for sentiment analysis.
        
        The orchestrator's job is to collect predictions from various models (GPT, Claude, Gemini, DeepSeek)
        and then make a final decision based on their collective input.
        
        Args:
            model_name (str): The GPT model we'll use as the orchestrator (e.g., 'gpt-5-2025-08-07')
            datasets_path (str): Where to find our datasets folder
            dataset_file (str): The CSV file we're working with
        """
        self.model_name = model_name
        self.datasets_path = datasets_path
        self.dataset_file = dataset_file
        self.current_row_index = None  # keeps track of which row we're currently processing
        
        # All the different AI models we're getting predictions from
        # These are the "agents" in our agentic system - each one analyzes the review independently
        self.prediction_columns = [
            'gemini-2.5-pro_prediction',
            'gemini-2.5-flash_prediction',
            'gemini-2.5-flash-lite_prediction',
            'gpt-5_prediction',
            'gpt-5-mini_prediction',
            'gpt-5-nano-2025-08-07_prediction',
            'gpt-4.1_prediction',
            'claude-sonnet-4-5_prediction',
            'claude-haiku-4-5_prediction',
            'claude-opus-4-1_prediction',
            'deepseek-chat_prediction',
            'deepseek-reasoner_prediction'
        ]
        
        # The prompt we send to the orchestrator model
        # This tells it how to interpret the agent predictions and make its final decision
        self.orchestrator_prompt_template = '''You are an orchestrator on an agentic ai sentiment analysis system for product reviews. Your task is to predict the star rating (1-5) that a user would give based on their review text.

Rating scale:
- 5 stars: Extremely positive, enthusiastic praise
- 4 stars: Positive with minor reservations
- 3 stars: Mixed feelings, balanced pros and cons
- 2 stars: Mostly negative with few positives
- 1 star: Extremely negative, strong dissatisfaction

Analyze the review's sentiment, key phrases, and overall tone. Consider:
- Emotional language (love, hate, disappointed, amazing)
- Specific praise or complaints
- Comparison words (better, worse, expected more)
- Problem severity mentions
- Recommendation likelihood

Your AI agents have already made their own predictions:

Your task is to assess their predictions based on the Review given and conclude to the appropriate prediction of the rating for this review.

Below I am providing you AI Agents' predictions:
{predictions}

Output only valid JSON with no additional text:
{{"rating": <number between 1-5>, "reasoning": <reason what you detect on ai agents' predictions, which agent deviates from the others and how you conclude to your rating prediction>}}

Review:
'''
        
        # Configuration for our orchestrator model
        # We're using GPTmethods here because the orchestrator is always a GPT model
        self.params = {
            'model_id': self.model_name,
            'prediction_column': f'{self.model_name}_agentic_prediction',
            'pre_path': self.datasets_path,
            'data_set': self.dataset_file,
            'prompt_array': {},
            'system': 'You are an expert orchestrator for an agentic AI sentiment analysis system.',
            'prompt': self.orchestrator_prompt_template,
            'feature_col': 'text',
            'label_col': 'rating',
            'json_key': 'rating',
            'max_tokens': 1500,
            'temperature': 0,
        }
        
        # Create the orchestrator instance
        self.gpt_orchestrator = GPTmethods(self.params)
        
        # Column name where we'll store the orchestrator's reasoning for its decisions
        # This helps us understand why it made certain choices
        self.reasoning_column = f'{self.model_name}_agentic_reasoning'
        
        # Set up the mapping between prediction columns and their model classes
        self._init_agent_configs()
    
    def _init_agent_configs(self):
        """
        Maps each prediction column to its model class and ID.
        
        It tells us which model class to instantiate
        and which specific model ID to use when we need to get predictions from each agent.
        We need this because different models use different API classes (Gemini, GPT, Claude, DeepSeek).
        """
        self.agent_configs = {
            'gemini-2.5-pro_prediction': {
                'class': GEMINImethods,
                'model_id': 'gemini-2.5-pro'
            },
            'gemini-2.5-flash_prediction': {
                'class': GEMINImethods,
                'model_id': 'gemini-2.5-flash'
            },
            'gemini-2.5-flash-lite_prediction': {
                'class': GEMINImethods,
                'model_id': 'gemini-2.5-flash-lite'
            },
            'gpt-5_prediction': {
                'class': GPTmethods,
                'model_id': 'gpt-5-2025-08-07'
            },
            'gpt-5-mini_prediction': {
                'class': GPTmethods,
                'model_id': 'gpt-5-mini-2025-08-07'
            },
            'gpt-5-nano-2025-08-07_prediction': {
                'class': GPTmethods,
                'model_id': 'gpt-5-nano-2025-08-07'
            },
            'gpt-4.1_prediction': {
                'class': GPTmethods,
                'model_id': 'gpt-4.1-2025-04-14'
            },
            'claude-sonnet-4-5_prediction': {
                'class': CLAUDEmethods,
                'model_id': 'claude-sonnet-4-5'
            },
            'claude-haiku-4-5_prediction': {
                'class': CLAUDEmethods,
                'model_id': 'claude-haiku-4-5'
            },
            'claude-opus-4-1_prediction': {
                'class': CLAUDEmethods,
                'model_id': 'claude-opus-4-1'
            },
            'deepseek-chat_prediction': {
                'class': DEEPSEEKmethods,
                'model_id': 'deepseek-chat'
            },
            'deepseek-reasoner_prediction': {
                'class': DEEPSEEKmethods,
                'model_id': 'deepseek-reasoner'
            }
        }
    
    def get_live_predictions(self, row):
        """
        Calls all 12 agent models to get their predictions for a single review.
        We're getting real-time predictions from each model.
        Each agent analyzes the review independently and gives us their rating prediction (1-5 stars).

        Args:
            row (pd.Series): The review data (contains title, text, etc.)
            
        Returns:
            dict: Maps each prediction column to its rating. 
                  e.g., {'gpt-5_prediction': 4, 'claude-sonnet_prediction': 5, ...}
        """
        predictions = {}
        
        # Base configuration that all agents will use
        # We keep it simple - just ask them to predict the rating from the review
        base_params = {
            'pre_path': self.datasets_path,
            'data_set': self.dataset_file,
            'prompt_array': {},
            'system': 'You are a sentiment analysis expert. Analyze product reviews and predict star ratings.',
            'prompt': '''Analyze this product review and predict the star rating (1-5) the user would give.

Rating scale:
- 5 stars: Extremely positive, enthusiastic praise
- 4 stars: Positive with minor reservations
- 3 stars: Mixed feelings, balanced pros and cons
- 2 stars: Mostly negative with few positives
- 1 star: Extremely negative, strong dissatisfaction

Output only valid JSON: {{"rating": <number between 1-5>}}

Review:
''',
            'feature_col': 'text',
            'label_col': 'rating',
            'json_key': 'rating',
            'max_tokens': 500,
            'temperature': 0,
        }
        
        print(f"\nGetting live predictions from all agents for row...")
        print("=" * 70)
        
        # Loop through each agent and get its prediction
        for pred_col in self.prediction_columns:
            if pred_col in self.agent_configs:
                config = self.agent_configs[pred_col]
                model_class = config['class']
                model_id = config['model_id']
                
                try:
                    # Set up parameters for this specific agent
                    agent_params = base_params.copy()
                    agent_params['model_id'] = model_id
                    agent_params['prediction_column'] = pred_col
                    
                    # Create an instance of the model class (GPTmethods, GEMINImethods, etc.)
                    agent = model_class(agent_params)
                    
                    print(f"Calling {pred_col} ({model_id})...", end=" ")
                    
                    # Each model class has its own prediction method (gpt_prediction, gemini_prediction, etc.)
                    # We need to call the right one based on the agent type
                    if isinstance(agent, GPTmethods):
                        result = agent.gpt_prediction(row)
                    elif isinstance(agent, GEMINImethods):
                        result = agent.gemini_prediction(row)
                    elif isinstance(agent, CLAUDEmethods):
                        result = agent.claude_prediction(row)
                    elif isinstance(agent, DEEPSEEKmethods):
                        result = agent.deepseek_prediction(row)
                    
                    # Check if we got a valid prediction
                    if result['status'] and result['data'].get(agent.json_key):
                        rating = result['data'][agent.json_key]
                        predictions[pred_col] = rating
                        print(f"✓ Predicted: {rating}")
                    else:
                        # Model call succeeded but didn't return a rating - that's weird
                        predictions[pred_col] = None
                        print(f"Failed")
                        logging.warning(f"Failed to get prediction from {pred_col}: {result}")
                        
                except Exception as e:
                    # Something went wrong (API error, network issue, etc.)
                    predictions[pred_col] = None
                    print(f"Error: {str(e)}")
                    logging.error(f"Error getting prediction from {pred_col}: {str(e)}")
            else:
                # This shouldn't happen unless we forgot to add a config for a prediction column
                predictions[pred_col] = None
                print(f"{pred_col}: No configuration found")
        
        print("=" * 70)
        return predictions
    
    def get_predictions_text(self, row, live_predictions=None):
        """
        Formats all agent predictions into a nice text block for the orchestrator.
        
        We either use freshly generated predictions (if we just got them from get_live_predictions)
        or fall back to existing predictions in the dataset row. This flexibility is useful
        for both live mode and analyzing existing results.
        
        Args:
            row (pd.Series): The dataset row (might have existing predictions)
            live_predictions (dict, optional): Fresh predictions we just got from agents
            
        Returns:
            str: Nicely formatted text like "gpt-5_prediction: 4\ngemini-2.5-pro_prediction: 5\n..."
        """
        predictions_text = ""
        
        for col in self.prediction_columns:
            # Prioritize live predictions if we have them, otherwise use what's in the dataset
            if live_predictions and col in live_predictions:
                pred_value = live_predictions[col]
                if pred_value is not None:
                    predictions_text += f"{col}: {pred_value}\n"
                else:
                    predictions_text += f"{col}: N/A\n"
            elif col in row.index and pd.notna(row[col]):
                predictions_text += f"{col}: {row[col]}\n"
            else:
                predictions_text += f"{col}: N/A\n"
        
        return predictions_text
    
    def generate_orchestrator_prompt(self, row, live_predictions=None):
        """
        Builds the complete prompt we'll send to the orchestrator.
        
        This combines three things:
        1. The base instructions (how to be an orchestrator)
        2. All the agent predictions
        3. The actual review text
        
        Args:
            row (pd.Series): The review data
            live_predictions (dict, optional): Fresh predictions from agents
            
        Returns:
            str: The full prompt ready to send to the orchestrator model
        """
        # Get all agent predictions formatted nicely
        predictions_text = self.get_predictions_text(row, live_predictions)
        
        # Insert the predictions into our template
        prompt = self.orchestrator_prompt_template.format(predictions=predictions_text)
        
        # Add the actual review at the end (title + text)
        review_text = ""
        if 'title' in row.index and pd.notna(row['title']):
            review_text += str(row['title']) + '\n'
        if 'text' in row.index and pd.notna(row['text']):
            review_text += str(row['text'])
        
        return prompt + review_text
    
    def run_orchestration(self):
        """
        The main method - processes the whole dataset with our agentic system.
        
        Here's how it works:
        1. For each review in the dataset:
           - Get predictions from all 12 agent models (live API calls)
           - Save those predictions to the dataset
           - Send all predictions to the orchestrator
           - Let the orchestrator make the final decision
           - Save the orchestrator's prediction and reasoning
        
        This can take a while since we're making 13 API calls per review (12 agents + orchestrator).
        
        Returns:
            dict: {'status': True/False, 'data': message about what happened}
        """
        print("=" * 70)
        print("Starting Agentic AI Orchestration")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.datasets_path}{self.dataset_file}")
        print(f"Prediction column: {self.params['prediction_column']}")
        print(f"Reasoning column: {self.reasoning_column}")
        print("=" * 70)
        
        # Load up our dataset
        try:
            import os
            df = pd.read_csv(self.datasets_path + self.dataset_file)
            print(f"\nLoaded dataset with {len(df)} rows")
            
            # Make sure we have a column for the orchestrator's reasoning
            if self.reasoning_column not in df.columns:
                df[self.reasoning_column] = pd.NA
                print(f"Added reasoning column: {self.reasoning_column}")
            
            # Create columns for all agent predictions if they don't exist yet
            # This way we can save agent predictions as we get them
            for col in self.prediction_columns:
                if col not in df.columns:
                    df[col] = pd.NA
                    print(f"Added column: {col}")
            
            # We'll store reasoning here as we process rows
            reasoning_data = {}
            
            # This will hold the live predictions for whichever row we're currently processing
            current_live_predictions = {}
            
            # We need to customize how the orchestrator makes predictions
            # Save the original method in case we need it later
            original_gpt_prediction = self.gpt_orchestrator.gpt_prediction
            
            def custom_gpt_prediction(input_row):
                """
                Our custom orchestrator prediction logic.
                
                This replaces the standard GPT prediction method so we can use our special
                orchestrator prompt that includes all the agent predictions.
                """
                try:
                    # Figure out which row we're working on
                    current_idx = self.current_row_index
                    
                    # Build the prompt with all agent predictions included
                    full_prompt = self.generate_orchestrator_prompt(df.loc[current_idx], current_live_predictions)
                    
                    # Print the prompt being sent
                    # print("\n" + "─" * 70)
                    # print(f"PROMPT FOR ROW {current_idx}:")
                    print("─" * 70)
                    # Show first 500 characters of prompt
                    # if len(full_prompt) > 500:
                    #     print(full_prompt[:500] + f"\n... [truncated, total length: {len(full_prompt)} chars]")
                    # else:
                    # print(full_prompt)
                    # print("─" * 70)
                    
                    # Set up the conversation for the orchestrator
                    # We don't actually use modified_input, but keeping it here for consistency
                    modified_input = {
                        'title': '',
                        self.gpt_orchestrator.feature_col: full_prompt
                    }
                    
                    # Format as a conversation (system message + user message)
                    conversation = [
                        {'role': 'system', 'content': self.gpt_orchestrator.system},
                        {'role': 'user', 'content': full_prompt}
                    ]
                    
                    # Call the orchestrator and retry if needed
                    import time
                    while True:
                        conversation_response = self.gpt_orchestrator.gpt_conversation(conversation)
                        
                        # Sometimes the API returns None - just retry if that happens
                        if conversation_response is None:
                            logging.warning("Received None response. Retrying...")
                            time.sleep(1)
                            continue
                        
                        # Get the actual text response
                        content = conversation_response.content
                        
                        # Print the raw response
                        # print(f"\nRAW RESPONSE FROM MODEL:")
                        # print("─" * 70)
                        # print(content)
                        # print("─" * 70)

                        # Parse the JSON response (should contain rating and reasoning)
                        import json
                        try:
                            # Most of the time the model returns pure JSON
                            json_data = json.loads(content)
                            cleaned_response = {"status": True, "data": json_data}
                        except json.JSONDecodeError:
                            # Sometimes the model wraps JSON in extra text, so we extract it
                            try:
                                start_idx = content.find('{')
                                end_idx = content.rfind('}')
                                if start_idx != -1 and end_idx != -1:
                                    json_str = content[start_idx:end_idx + 1]
                                    json_data = json.loads(json_str)
                                    cleaned_response = {"status": True, "data": json_data}
                                else:
                                    cleaned_response = {"status": False, "data": f"No JSON found in response: {content}"}
                            except Exception as e:
                                cleaned_response = {"status": False, "data": f"JSON parsing error: {str(e)}"}
                        
                        # Check if we successfully parsed the JSON
                        if cleaned_response["status"]:
                            # Print it nicely formatted
                            print(json.dumps(cleaned_response['data'], indent=2))
                            
                            # Save the reasoning text if the orchestrator provided it
                            if 'reasoning' in cleaned_response['data']:
                                reasoning_data[current_idx] = cleaned_response['data']['reasoning']
                            
                            return cleaned_response
                        else:
                            # Parsing failed - show the error and retry
                            print(f"\nFAILED TO PARSE JSON:")
                            print("─" * 70)
                            print(f"Error: {cleaned_response['data']}")
                            print("─" * 70)
                        
                        logging.warning("Invalid response format. Retrying...")
                        time.sleep(1)
                        
                except Exception as e:
                    # Something went really wrong - log it and return an error
                    logging.error(f"Error in custom_gpt_prediction for row {self.current_row_index}: {str(e)}")
                    print(f"\nEXCEPTION in custom_gpt_prediction:")
                    print("─" * 70)
                    print(f"{str(e)}")
                    print("─" * 70)
                    return {"status": False, "data": str(e)}
            
            # Swap out the default prediction method with our custom one
            self.gpt_orchestrator.gpt_prediction = custom_gpt_prediction
            
            # Now let's process the dataset!
            print(f"\nProcessing predictions...")
            prediction_col = self.params['prediction_column']
            time_col = "time-" + self.model_name
            
            # Make sure we have columns for the orchestrator's predictions and timing
            if prediction_col not in df.columns:
                df[prediction_col] = pd.NA
            if time_col not in df.columns:
                df[time_col] = pd.NA
            
            # Save the structure so we can see the new columns
            df.to_csv(self.datasets_path + self.dataset_file, index=False)
            
            # Go through each row and process it if it doesn't have an orchestrator prediction yet
            for index, row in df.iterrows():
                if pd.isnull(row[prediction_col]):
                    self.current_row_index = index
                    
                    print(f"\n{'='*70}")
                    print(f"Processing Row {index}")
                    print(f"{'='*70}")
                    
                    start_time = time.time()
                    
                    # STEP 1: Get predictions from all 12 agents
                    # This makes live API calls to GPT, Claude, Gemini, and DeepSeek models
                    print(f"\nStep 1: Getting predictions from all agent models...")
                    current_live_predictions = self.get_live_predictions(row)
                    
                    # Save all those predictions to the dataset right away
                    for pred_col, pred_value in current_live_predictions.items():
                        if pred_value is not None:
                            df.at[index, pred_col] = pred_value
                    
                    # Write to CSV so we don't lose data if something crashes
                    df.to_csv(self.datasets_path + self.dataset_file, index=False)
                    print(f"\n✓ Agent predictions saved to dataset")
                    
                    # STEP 2: Now let the orchestrator analyze all the agent predictions
                    print(f"\nStep 2: Running orchestrator to make final prediction...")
                    prediction = custom_gpt_prediction(row)
                    
                    end_time = time.time()
                    elapsed_time = round(end_time - start_time, 4)
                    
                    if not prediction['status']:
                        # Orchestrator failed - stop processing and report the error
                        print(f"\nRow {index}: Prediction failed - {prediction['data']}")
                        logging.error(f"Row {index}: Prediction failed - {prediction['data']}")
                        break
                    else:
                        # Success! Extract the orchestrator's rating
                        if prediction['data'].get(self.gpt_orchestrator.json_key):
                            rating = prediction['data'][self.gpt_orchestrator.json_key]
                            
                            # Save the orchestrator's prediction and how long it took
                            df.at[index, prediction_col] = rating
                            df.at[index, time_col] = elapsed_time
                            
                            # Also save the reasoning if the orchestrator provided it
                            if index in reasoning_data:
                                df.at[index, self.reasoning_column] = reasoning_data[index]
                            
                            # Write everything to the CSV (we do this after each row to avoid data loss)
                            df.to_csv(self.datasets_path + self.dataset_file, index=False)
                            
                            # Print a nice summary of what just happened
                            print(f"\n{'='*70}")
                            print(f"SAVED TO Dataset.csv - Row {index}")
                            print(f"{'='*70}")
                            print(f"  Prediction: {rating} stars")
                            print(f"  Time: {elapsed_time}s")
                            if index in reasoning_data:
                                reasoning_text = reasoning_data[index]
                                # Truncate long reasoning to keep output readable
                                if len(reasoning_text) > 150:
                                    print(f"  Reasoning: {reasoning_text[:150]}...")
                                else:
                                    print(f"  Reasoning: {reasoning_text}")
                            print(f"{'='*70}\n")
                        else:
                            # Weird - we got a successful response but no rating in it
                            error_msg = f"No rating found in prediction for row {index}"
                            logging.error(error_msg)
                            print(f"\n{error_msg}")
                            return {"status": False, "data": error_msg}
            
            # All done! Print a final summary
            print("=" * 70)
            print("Agentic AI Orchestration Complete!")
            print(f"Predictions stored in: {self.datasets_path}{self.dataset_file}")
            print(f"  - Prediction column: {prediction_col}")
            print(f"  - Time column: {time_col}")
            print(f"  - Reasoning column: {self.reasoning_column}")
            print("=" * 70)
            
            return {"status": True, "data": "Predictions have successfully been completed"}
            
        except Exception as e:
            # Top-level error handler - catch anything that went wrong
            error_msg = f"Error during orchestration: {str(e)}"
            logging.error(error_msg)
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "status": False,
                "data": error_msg
            }


# Set up error logging so we can debug issues later
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)


if __name__ == "__main__":
    # Set up the orchestrator with the model we want to use
    model_name = 'gpt-5-2025-08-07'
    # model_name = 'gpt-5-mini-2025-08-07'  # uncomment to use the mini model instead
    
    orchestrator = AgenticAI(
        model_name=model_name,
        datasets_path='Datasets/',
        dataset_file='Dataset.csv'
    )
    
    # Start the orchestration process
    result = orchestrator.run_orchestration()
    
    # Check how it went
    if result['status']:
        print("\nSuccess!")
    else:
        print(f"\nFailed: {result['data']}")
