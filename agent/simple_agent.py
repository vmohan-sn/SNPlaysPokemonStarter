import base64
import collections
import copy
import io
import json # Added for parsing suggested buttons
import logging
import os
import re # Added for parsing suggested buttons

# Updated config import
from config import (
    MAX_TOKENS, TEMPERATURE, USE_NAVIGATOR,
    VISION_MODEL_NAME, ACTION_MODEL_NAME,
    SAMBANOVA_API_KEY, SAMBANOVA_BASE_URL
)

from agent.emulator import Emulator
# Removed: from anthropic import Anthropic
from openai import OpenAI # Added

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM_PROMPT = """You are an expert Pokemon Red strategist. Your goal is to help the player make progress by defining immediate objectives and suggesting how to achieve them.

Based on the current game observation (visuals, memory, map), you must:
1.  Define the most critical immediate short-term objective. Be very specific (e.g., "My immediate objective is to exit this room through the door visible at the top of the screen." or "My immediate objective is to talk to the person standing in front of me.").
2.  Briefly explain how this short-term objective contributes to long-term game progress (e.g., "This is necessary to start exploring Pallet Town and eventually receive my first Pokemon." or "This person might give me a crucial item or information to advance the story.").
3.  Propose a specific sequence of button presses that you believe will achieve this immediate short-term objective. The button presses MUST be formatted as a JSON list of strings on a line that starts with 'Suggested Buttons: '. For example: `Suggested Buttons: ["up", "up", "a"]`.

Your entire response should be a single text block.

Example response:
My immediate objective is to exit this room through the door visible at the top of the screen. This is the first step to explore Pallet Town and begin my Pokemon journey.
Suggested Buttons: ["up", "up", "up"]"""

def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your primary goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen and the information from your game memory.

**Strategist Input and Your Role:**
You will be provided with strategic guidance (a textual description of a short-term objective and its relevance) and potentially a *suggested button sequence* from a planning assistant (Strategist LLM). Your task is to critically analyze the current game state (screen, memory, map), the textual strategic guidance, and any suggested button sequence.

Your decision-making process before each action should be:
1.  **Understand the Goal:** Review the textual strategic guidance to understand the immediate short-term objective.
2.  **Evaluate Suggested Buttons:** If a button sequence is suggested by the Strategist, assess its appropriateness and effectiveness in achieving the stated objective, considering the current visual information, memory, and map.
3.  **Decide on Action:**
    *   If the suggested buttons are good and directly help achieve the stated short-term goal, you should generally use them.
    *   However, you have the authority to *deviate* from, *correct*, or *replace* the suggested buttons if they seem incorrect, inefficient, conflict with the current game state, or if you identify a better sequence to achieve the goal. Your own assessment of the live game state takes precedence.
4.  **Explain Reasoning:** Briefly explain your reasoning for the chosen action, including why you are following, modifying, or disregarding any suggested buttons.
5.  **Execute Action:** Your final output must be a tool call for the chosen button presses, using the 'press_buttons' tool (or 'navigate_to' if appropriate and available).

**Self-Correction and Learning:** If you notice your recent actions are not leading to meaningful progress (e.g., the environment doesn't change, you're repeatedly blocked, or you're not achieving your immediate objective), actively try different strategies. Re-evaluate the visual information, collision map, memory data, and the Strategist's advice. If one approach isn't working, try alternative button presses, exploring different directions, or interacting with different objects.

Early in the game (like when you first start or are in a new building), your objective is often to explore your immediate surroundings and find a way to the next area. This might involve looking for doors, stairs, or paths leading outwards. Pay attention to the 'Valid Moves' information from your memory, as it indicates directions you can immediately move.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "parameters": { # Formerly input_schema
                "type": "object",
                "properties": {
                    "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                }
            },
            "required": ["buttons"],
            }
        }
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append({
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
            "parameters": { # Formerly input_schema
                "type": "object",
                "properties": {
                    "row": {
                    "type": "integer",
                    "description": "The row coordinate to navigate to (0-8)."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate to navigate to (0-9)."
                }
            },
            "required": ["row", "col"],
            }
        }
    })


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        # Updated client initialization
        self.client = OpenAI(
            base_url=SAMBANOVA_BASE_URL,
            api_key=SAMBANOVA_API_KEY
        )
        self.running = True
        # Initial message content is a string.
        self.message_history = [{"role": "user", "content": "You may now begin playing. Describe what you see and what you plan to do."}]
        self.max_history = max_history
        self.image_history = collections.deque(maxlen=3)
        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)

    def _transform_messages_for_action_model(self, messages_history: list) -> list:
        """
        Transforms messages for ACTION_MODEL_NAME:
        - User/System messages: Ensures 'content' is a string.
        - Assistant/Tool messages: Kept as is.
        """
        transformed_messages = []
        for msg in messages_history:
            role = msg.get("role")
            content = msg.get("content")

            if role in ["user", "system"]:
                if isinstance(content, list):
                    text_content_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                            text_content_parts.append(item["text"])
                        elif isinstance(item, str): # New condition
                            text_content_parts.append(item)
                        # Else: item is neither a text dict nor a string, potentially log or ignore
                    
                    if not text_content_parts:
                        logger.warning(f"User/System message content list {content} yielded no text parts.")
                        # Fallback to string representation of the original list, though this path should be less likely now.
                        final_text_content = str(content) 
                    else:
                        # Join the parts. Use "\n" if multiple parts, otherwise just the single part.
                        final_text_content = "\n".join(text_content_parts).strip()

                    transformed_messages.append({"role": role, "content": final_text_content})
                elif isinstance(content, str):
                    transformed_messages.append({"role": role, "content": content})
                else:
                    logger.warning(f"Unexpected content type for role {role}: {type(content)}. Converting to string.")
                    transformed_messages.append({"role": role, "content": str(content)})
            else: # assistant, tool, or other roles
                transformed_messages.append(msg) # Keep as is
        return transformed_messages

    def get_visual_context_for_action_model(self, screenshots_b64: list[str], memory_info: str, collision_map_str: str = None) -> str:
        """
        Calls the VISION_MODEL_NAME to get a textual description of the game state including screen, memory, and collision map.
        """
        logger.info(f"Calling Vision Model ({VISION_MODEL_NAME}) to get visual context...")
        
        vision_system_prompt = "You are an expert at analyzing game screenshots and memory data for a Pokemon game. You will receive up to three sequential game screenshots (oldest to newest), memory data, and optionally a collision map.\nYour task is to:\n1.  Describe the general type of environment (e.g., 'small room', 'hallway', 'outdoors', 'cave').\n2.  Identify and describe any potential exits from the current view, such as doors, doorways, stairs, or paths leading off-screen. Specify their apparent direction (e.g., 'door to the south', 'stairs going up on the east side', 'path leading west').\n3.  Describe other key visual elements, player status, and any noticeable changes or movement across the frames.\n4.  Incorporate any relevant information from the memory data that would be useful for deciding the next game action.\n5.  If a collision map is provided, use it to confirm visible paths or barriers, and mention any obvious paths based on it, especially if they align with potential exits.\nFocus on what the player character sees, can interact with, or might use to navigate. Be concise but thorough."
        
        user_content = [{"type": "text", "text": "Here are the recent game screenshots (oldest to newest), memory data, and optionally a collision map:"}]
        
        image_list_for_api = list(screenshots_b64) # screenshots_b64 is a deque, so this is [oldest, ..., newest]
        
        if len(image_list_for_api) == 3:
            labels = ["Oldest game screenshot:", "Previous game screenshot:", "Latest game screenshot:"]
        elif len(image_list_for_api) == 2:
            labels = ["Older game screenshot:", "Latest game screenshot:"]
        elif len(image_list_for_api) == 1:
            labels = ["Current game screenshot:"]
        else:
            labels = [] # Should ideally not happen

        for i, b64_data in enumerate(image_list_for_api):
            if i < len(labels):
                user_content.append({"type": "text", "text": labels[i]})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
            })
            
        user_content.append({"type": "text", "text": f"Memory data: {memory_info}"})
        if collision_map_str:
            user_content.append({"type": "text", "text": f"Collision Map (player is 'P', obstacles are 'X', free space is '.'):\n{collision_map_str}"})

        messages = [
            {"role": "system", "content": vision_system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS, # Max tokens for the vision model's description
                temperature=TEMPERATURE # Can use the global temperature or a specific one for vision
            )
            
            description = response.choices[0].message.content if response.choices[0].message.content else ""
            if response.usage:
                 logger.info(f"Vision model usage: Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}")
            logger.info(f"Vision model description: {description}")
            return description if description.strip() else "Vision model returned an empty description."

        except Exception as e:
            logger.error(f"Error calling Vision Model ({VISION_MODEL_NAME}): {e}", exc_info=True)
            return f"Error: Could not get visual context from Vision Model. Details: {str(e)}"

    def process_tool_call(self, tool_call):
        """
        Process a single tool call.
        Returns only the tool_result_string.
        Screenshot and memory info are handled by the run method before calling the action model.
        """
        tool_name = tool_call.function.name 
        try:
            tool_input_str = tool_call.function.arguments
            import json 
            tool_input = json.loads(tool_input_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError for tool '{tool_name}' arguments: '{tool_input_str}'. Error: {e}")
            return f"Error: Could not parse JSON arguments for tool {tool_name}. Input: {tool_input_str}" # Return only string
        except Exception as e: 
            logger.error(f"Error processing tool '{tool_name}' arguments: {tool_input_str}. Error: {e}")
            return f"Error: Exception processing arguments for tool {tool_name}. Input: {tool_input_str}" # Return only string

        logger.info(f"Processing tool call: {tool_name} with input {tool_input}")
        tool_result_string = ""
        
        # Execute action first
        if tool_name == "press_buttons":
            buttons = tool_input.get("buttons", [])
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")
            self.emulator.press_buttons(buttons, wait) 
            tool_result_string = f"Pressed buttons: {', '.join(buttons)}."
        elif tool_name == "navigate_to" and USE_NAVIGATOR: 
            row = tool_input.get("row")
            col = tool_input.get("col")
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")
            status, path = self.emulator.find_path(row, col) 
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                tool_result_string = f"Navigation successful: followed path with {len(path)} steps to ({row},{col})."
            else:
                tool_result_string = f"Navigation failed: {status} for target ({row},{col})."
        else:
            logger.error(f"Unknown or unavailable tool called: {tool_name}")
            tool_result_string = f"Error: Unknown or unavailable tool '{tool_name}' called."
            
        # Get fresh screenshot and memory info AFTER the action, but they are not returned here.
        # The 'run' method will call emulator again for the next turn's visual context.
        # This might be slightly inefficient but simplifies data flow.
        # For now, just log them here if needed for debugging.
        screenshot_after_action = self.emulator.get_screenshot()
        memory_info_after_action = self.emulator.get_state_from_memory()
        collision_map_after_action = self.emulator.get_collision_map()
        collision_map_str_after_action = f"\n{collision_map_after_action}" if collision_map_after_action else ""
        logger.info(f"[Memory State after action in process_tool_call]\n{memory_info_after_action}")
        if collision_map_str_after_action:
            logger.info(f"[Collision Map after action in process_tool_call]{collision_map_str_after_action}")

        return tool_result_string # Return only the string summary

    def run(self, num_steps=1):
        """Main agent loop.
        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")
        steps_completed = 0

        # Prepare initial context if this is the very first message
        if len(self.message_history) == 1 and \
           self.message_history[0]["role"] == "user" and \
           isinstance(self.message_history[0]["content"], list) and \
           self.message_history[0]["content"][0].get("type") == "text" and \
           self.message_history[0]["content"][0].get("text", "").startswith("You may now begin playing"):
            
            logger.info("Augmenting initial message with visual context...")
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
            self.image_history.append(screenshot_b64)
            memory_info = self.emulator.get_state_from_memory()
            collision_map = self.emulator.get_collision_map()
            collision_map_str = f"\n{collision_map}" if collision_map else ""
            
            initial_visual_description = self.get_visual_context_for_action_model(list(self.image_history), memory_info, collision_map_str)
            
            # Augment the initial user message string with the first visual description.
            self.message_history[0]["content"] += f"\n\nInitial Observation:\n{initial_visual_description}"

        while self.running and steps_completed < num_steps:
            try:
                # Construct the user message string for the current turn
                user_content_parts = []

                # Construct the user message string for the current turn
                user_content_parts = []

                # If the last message was a tool result, prepend its content and a reflection cue.
                if self.message_history and self.message_history[-1]["role"] == "tool":
                    last_tool_message = self.message_history.pop() 
                    tool_name = last_tool_message.get("name", "Unknown tool")
                    tool_output_string = last_tool_message.get("content", "No content from tool.")
                    # New formulation for previous action context:
                    action_feedback_prompt = (
                        f"Your previous action was '{tool_name}' with result: '{tool_output_string}'.\n"
                        f"Now, observe the current situation. If this action did not result in meaningful progress "
                        f"or the environment seems unchanged, reassess your strategy and consider alternative actions or exploring different options."
                    )
                    user_content_parts.append(action_feedback_prompt)
                
                # Get current visual and memory context
                current_screenshot = self.emulator.get_screenshot()
                current_screenshot_b64 = get_screenshot_base64(current_screenshot, upscale=2)
                self.image_history.append(current_screenshot_b64)
                current_memory_info = self.emulator.get_state_from_memory() # String from emulator
                current_collision_map = self.emulator.get_collision_map()
                current_collision_map_str = f"\n{current_collision_map}" if current_collision_map else ""
                
                visual_description = self.get_visual_context_for_action_model(
                    list(self.image_history), 
                    current_memory_info,
                    current_collision_map_str # Pass it to vision model as before
                )

                # --- Begin Strategist LLM Call ---
                strategic_suggestion = "No strategic suggestion obtained." # Default fallback
                try:
                    logger.info(f"Calling Strategist LLM ({ACTION_MODEL_NAME}) for strategic suggestion...")
                    strategist_user_message_content = (
                        f"Current Observation (from Vision Model):\n{visual_description}\n\n"
                        f"Relevant Game Memory:\n{current_memory_info}\n\n"
                    )
                    if current_collision_map_str:
                        strategist_user_message_content += f"Current Collision Map:\n{current_collision_map_str}\n\n"
                    
                    strategist_user_message_content += "Based on this, what is your strategic suggestion?"

                    strategist_messages = [
                        {"role": "system", "content": STRATEGIST_SYSTEM_PROMPT},
                        {"role": "user", "content": strategist_user_message_content}
                    ]
                    
                    # Transform messages for the Strategist LLM call
                    transformed_strategist_messages = self._transform_messages_for_action_model(strategist_messages)

                    strategist_response = self.client.chat.completions.create(
                        model=ACTION_MODEL_NAME,
                        messages=transformed_strategist_messages,
                        max_tokens=MAX_TOKENS, # Or a more modest number if suggestions are short
                        temperature=TEMPERATURE,
                        # tools=None, # Explicitly no tools for strategist
                        # tool_choice=None 
                    )
                    
                    if strategist_response.choices and strategist_response.choices[0].message.content:
                        strategic_suggestion = strategist_response.choices[0].message.content.strip()
                        logger.info(f"Strategic Suggestion: {strategic_suggestion}")
                        if strategist_response.usage:
                            logger.info(f"Strategist LLM usage: Input tokens: {strategist_response.usage.prompt_tokens}, Output tokens: {strategist_response.usage.completion_tokens}")
                    else:
                        logger.warning("Strategist LLM returned no content or empty choices.")
                
                except Exception as e:
                    logger.error(f"Error calling Strategist LLM ({ACTION_MODEL_NAME}): {e}", exc_info=True)
                    strategic_suggestion = "Error obtaining strategic suggestion." # Fallback on error
                # --- End Strategist LLM Call ---

                # --- Parse Strategist LLM Output ---
                textual_strategic_advice = strategic_suggestion # Default to the full suggestion
                parsed_suggested_buttons = [] # Default to empty list

                # Try to find the 'Suggested Buttons:' line and parse it
                # Regex to find 'Suggested Buttons: [...]' case-insensitively and multiline
                # It captures the JSON part for parsing.
                button_match = re.search(r"^\s*Suggested Buttons:\s*(\[.*\])\s*$", strategic_suggestion, re.MULTILINE | re.IGNORECASE)
                
                if button_match:
                    buttons_json_str = button_match.group(1)
                    try:
                        parsed_suggested_buttons = json.loads(buttons_json_str)
                        # Validate that it's a list of strings, if not, revert to empty
                        if not (isinstance(parsed_suggested_buttons, list) and \
                                all(isinstance(btn, str) for btn in parsed_suggested_buttons)):
                            logger.warning(f"Parsed buttons are not a list of strings: {parsed_suggested_buttons}. Reverting to empty list.")
                            parsed_suggested_buttons = []
                        else:
                            logger.info(f"Successfully parsed suggested buttons: {parsed_suggested_buttons}")
                            # Remove the button line from the textual advice for clarity
                            textual_strategic_advice = strategic_suggestion.replace(button_match.group(0), "").strip()
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON for suggested buttons: {buttons_json_str}. Keeping full text as advice.")
                        # parsed_suggested_buttons remains []
                else:
                    logger.info("No 'Suggested Buttons:' line found in strategist output. Full text is advice.")
                
                logger.info(f"Textual Strategic Advice: {textual_strategic_advice}")
                logger.info(f"Parsed Suggested Buttons: {parsed_suggested_buttons}")
                # --- End Parsing Strategist LLM Output ---
                
                user_content_parts.append(f"Current Observation (from Vision Model):\n{visual_description}")
                if current_collision_map_str: # Add collision map directly if it exists
                    user_content_parts.append(f"Current Collision Map:\n{current_collision_map_str}")
                user_content_parts.append(f"Relevant Game Memory:\n{current_memory_info}")
                
                # Incorporate the parsed Strategist LLM output into the user content for the Tactical Action LLM
                user_content_parts.append(f"Strategic Guidance (from Assistant Strategist):\n{textual_strategic_advice}")

                if parsed_suggested_buttons: # Only add if buttons were successfully parsed and are not empty
                    user_content_parts.append(f"Strategist's Suggested Buttons: {str(parsed_suggested_buttons)}")
                else:
                    user_content_parts.append("Strategist's Suggested Buttons: None provided or parsing failed.")
                
                user_content_parts.append(
                    "Now, considering all the above (previous action, current observation, memory, strategic guidance, and any suggested buttons), "
                    "explain your reasoning and choose your next action (e.g., button presses)."
                )

                final_user_content_string = "\n\n".join(user_content_parts)

                # Update message history for the API call
                # The history sent to the API should reflect the latest state.
                messages_for_api = copy.deepcopy(self.message_history)

                # If the last message in messages_for_api is a user message (e.g. the initial one, or from summary),
                # and we are not on the very first step where initial augmentation happened,
                # then append the new context to it. Otherwise, add a new user message.
                is_initial_turn_with_augmented_message = (steps_completed == 0 and \
                                                       len(messages_for_api) == 1 and \
                                                       messages_for_api[0]["role"] == "user" and \
                                                       "Initial Observation" in messages_for_api[0]["content"])
                
                if not messages_for_api or messages_for_api[-1]["role"] != "user":
                    messages_for_api.append({"role": "user", "content": final_user_content_string})
                elif not is_initial_turn_with_augmented_message : 
                    # Last message is 'user', but it's not the initial one that was just augmented.
                    # This could be a user message from a summary, or if a turn somehow didn't add one.
                    # Append new context to existing user message content.
                     messages_for_api[-1]["content"] += f"\n\n{final_user_content_string}"
                # If it IS the initial_turn_with_augmented_message, its content is already set from pre-loop.
                # final_user_content_string was constructed based on the latest state,
                # and the initial message in self.message_history was already updated.
                # So, messages_for_api (a copy of self.message_history) is up-to-date for the first turn.
                # For subsequent turns, this logic correctly appends or adds new user message.

                # messages_for_api is the history built so far (already deepcopied and potentially modified)
                # Now, transform it specifically for the action model
                transformed_api_messages = self._transform_messages_for_action_model(messages_for_api)
                
                payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + transformed_api_messages
                
                logger.info(f"Calling {ACTION_MODEL_NAME}. Last user message content: {payload_messages[-1]['content'][:250] if payload_messages and payload_messages[-1]['role'] == 'user' else 'N/A...'}")
                response = self.client.chat.completions.create(
                    model=ACTION_MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    messages=payload_messages,
                    tools=AVAILABLE_TOOLS,
                    tool_choice="required", 
                    temperature=TEMPERATURE,
                )

                logger.info(f"Full API response object: {response}")

                # ADD THIS BLOCK:
                if response is None or not response.choices: # Checks if response itself is None OR if choices is None or empty
                    logger.warning(
                        "Received an empty or invalid response from action model (likely context length issue). "
                        "Triggering summarization and retrying turn."
                    )
                    if len(self.message_history) <= 1: # Avoid summarizing if history is already minimal
                        logger.error("History is too short to summarize further. Stopping agent to prevent loop.")
                        self.running = False
                    else:
                        try:
                            self.summarize_history()
                        except Exception as e:
                            logger.error(f"Error during summarization attempt: {e}. Stopping agent.")
                            self.running = False
                    continue # Skip the rest of this turn and try again
                # END OF ADDED BLOCK
                
                response_message = response.choices[0].message
                if response.usage: 
                    logger.info(f"Response usage: Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}")

                # Process response: assistant text and tool calls
                # Assistant message content parts (text or tool_call dicts)
                assistant_response_content_for_history = []
                if response_message.content: 
                    logger.info(f"[Assistant Text] {response_message.content}")
                    # For DeepSeek, assistant content for history should also be simple string if no tools.
                    # However, if there are tool_calls, OpenAI standard is list of parts.
                    # For now, let's assume DeepSeek handles list if tool_calls are present, or string if only text.
                    # The subtask implies string for USER and SYSTEM. Assistant messages are handled by OpenAI library.
                    # The important part is that what WE construct for user/system is string.
                    assistant_response_content_for_history.append({"type": "text", "text": response_message.content})
                
                tool_calls_from_response = response_message.tool_calls

                if tool_calls_from_response:
                    for tool_call_obj in tool_calls_from_response: 
                        logger.info(f"[Assistant Tool Call] ID: {tool_call_obj.id}, Function: {tool_call_obj.function.name}, Args: {tool_call_obj.function.arguments}")
                        assistant_response_content_for_history.append({
                            "type": "tool_call", # This is our internal representation
                            "id": tool_call_obj.id,
                            "function": { # Mimicking OpenAI's structure for consistency
                                "name": tool_call_obj.function.name,
                                "arguments": tool_call_obj.function.arguments 
                            }
                        })
                
                if assistant_response_content_for_history: 
                    self.message_history.append({ # Add assistant's turn to history
                        "role": "assistant",
                        "content": f"{assistant_response_content_for_history}" # This can be a list of text/tool_call parts
                    })
                
                if tool_calls_from_response:
                    for tool_call_obj in tool_calls_from_response: 
                        tool_result_string = self.process_tool_call(tool_call_obj) 
                        
                        self.message_history.append({ # Add tool result to history
                            "role": "tool", 
                            "tool_call_id": tool_call_obj.id, 
                            "name": tool_call_obj.function.name, 
                            "content": tool_result_string, # This is already a string
                        })
                # The next iteration will construct a new user message with latest observations + this tool result.

                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True) 
                self.running = False 

        if not self.running:
            self.emulator.stop()
        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info(f"[Agent] Generating conversation summary (current history length: {len(self.message_history)})...")
        
        # Get current visual context as text description for the summary
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
        self.image_history.append(screenshot_b64) # Add latest screenshot to history
        memory_info = self.emulator.get_state_from_memory()
        collision_map = self.emulator.get_collision_map()
        collision_map_str = f"\n{collision_map}" if collision_map else ""
        current_visual_text_description = self.get_visual_context_for_action_model(list(self.image_history), memory_info, collision_map_str)
        
        summary_request_messages = copy.deepcopy(self.message_history)
        summary_request_messages.append({ 
            "role": "user", # This will be transformed to string content by the helper
            "content": [{"type": "text", "text": SUMMARY_PROMPT}], 
        })
        
        # Transform messages for the summarization call as well
        transformed_summary_messages = self._transform_messages_for_action_model(summary_request_messages)
        payload_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + transformed_summary_messages
        
        response = self.client.chat.completions.create( 
            model=ACTION_MODEL_NAME, 
            max_tokens=MAX_TOKENS, 
            messages=payload_messages, # Pass transformed messages
            temperature=TEMPERATURE
        )
        
        summary_text = "".join([choice.message.content for choice in response.choices if choice.message.content])
        
        logger.info(f"[Agent] Game Progress Summary:\n{summary_text}")
        
        new_user_message_content = (
            f"CONVERSATION HISTORY SUMMARY (representing up to {self.max_history} previous messages):\n{summary_text}"
            f"\n\nLatest Game Context (after summary):\n{current_visual_text_description}"
            "\n\nYou were just asked to summarize your playthrough. The summary and current game context are above. Continue playing."
        )
        self.message_history = [
            {
                "role": "user",
                "content": new_user_message_content # CONTENT IS NOW THE STRING ITSELF
            }
        ]
        logger.info(f"[Agent] Message history condensed. New length: {len(self.message_history)}")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        if self.emulator: # Check if emulator exists
            self.emulator.stop()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Default ROM path relative to this script (agent/simple_agent.py)
    rom_path_default = os.path.join(current_dir, "..", "pokemon.gb") # Assumes script is in agent/, rom is in parent dir (e.g. /pokemon.gb)

    # Allow overriding ROM path with an environment variable for flexibility
    rom_path_env = os.environ.get("POKEMON_ROM_PATH")
    rom_path = rom_path_env if rom_path_env else rom_path_default

    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found at {rom_path}. Please set POKEMON_ROM_PATH or place it at the default location.")
        exit(1)
    logger.info(f"Using ROM file at {rom_path}")

    agent = SimpleAgent(rom_path, headless=True, max_history=20) # max_history for testing

    try:
        steps_to_run = 3 # Short run for testing
        logger.info(f"Attempting to run agent for {steps_to_run} steps...")
        steps_completed = agent.run(num_steps=steps_to_run)
        logger.info(f"Agent completed {steps_completed} steps.")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping main execution.")
    except Exception as e:
        logger.error(f"Unhandled error in main execution: {e}", exc_info=True)
    finally:
        logger.info("Stopping agent from main block.")
        agent.stop()
        logger.info("Agent stopped.")