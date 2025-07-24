
import logging
import inspect
import traceback
import re
from typing import Generator, Dict, Any, Union, List, Tuple
from vertexai.generative_models import (
    FunctionDeclaration,
    Tool,
    GenerationConfig,
    Part,
    GenerativeModel
)
from config import DEFAULT_MODEL
from gemini_agents_toolkit import scheduler
=======
import uuid 

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.runners import Runner

from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions

import threading


class TooManyFunctionCallsException(Exception):
    def __init__(self, message: str, call_history: list):
=======
    def __init__(self,message):
        super().__init__(message)

def log_retry_error(retry_state):
    """Logs the retry attempt details including the error message."""
    print(f"Retrying {retry_state.fn.__name__}... Attempt #{retry_state.attempt_number}, "
          f"Last error: {retry_state.outcome.exception()}")

class GeminiAgent:
=======
    logging.error(f"Retrying {retry_state.fn.__name__}... Attempt #{retry_state.attempt_number}, "
                  f"Last error: {retry_state.outcome.exception()}")


class ADKAgentService:
    """An agent to request LLM for executing users instructions with tools (custom functions) provided by user"""

    def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            *,
            functions: List[callable] = None,
            function_call_limit_per_chat: int = None,
            system_instruction: str = None,
            delegation_function_prompt: str = None,
            delegates: List['GeminiAgent'] = None,
            debug: bool = False,
            on_message: callable = None,
            generation_config: GenerationConfig = None
    ):
        if functions is None:
            functions = []
        self.history = []
        self.functions = {func.__name__: func for func in functions}
        func_declarations = [_generate_function_declaration(func) for func in functions]
        
        if delegates is not None:
            for i, delegate in enumerate(delegates):
                if not isinstance(delegate, GeminiAgent):
                    raise ValueError("Delegates must be a list of GeminiAgent instances")
                if not delegate.delegation_function_prompt:
                    raise ValueError("All delegates must have a delegation prompt specified")

                tool_name = f"delegate_{i}_send_message"
                self.functions[tool_name] = delegate.send_message
                func_declarations.append(
                    _generate_function_declaration(
                        delegate.send_message,
                        user_set_name=tool_name,
                        user_set_description=delegate.delegation_function_prompt))

        tools = [Tool(function_declarations=func_declarations)] if func_declarations else None
        self._model = GenerativeModel(
            model_name=model_name,
            tools=tools,
            system_instruction=system_instruction,
            generation_config=generation_config
        )
        self.chat = self._model.start_chat()
        self.debug = debug
        self.function_call_limit_per_chat = function_call_limit_per_chat
        self.on_message = on_message
        self.delegation_function_prompt = self._format_delegation_prompt(delegation_function_prompt)

    def _format_delegation_prompt(self, prompt: str) -> str:
        """Formats the delegation prompt if provided."""
        if prompt is not None:
            return f"""
            Use this function to delegate task to the agent (your coworker). 
            For delegate pass natural language command to the agent via this method.
            Here is what this agent can be delegate to do: {prompt}

            Keep in mind that the agent does not have full context of the conversation,
            you have to construct the command in a way that agent can understand
            and execute the task.
            The agent will treat the message you send as if it comes from user
            and will respond to it as if it is a user message."""
        return None

    def stream_response(
        self,
        prompt: str,
        *,
        generation_config: GenerationConfig = None,
        history: List[Dict] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream responses chunk-by-chunk while maintaining all agent functionality."""
        if history:
            self.set_history(history)

        try:
            response = self._model.generate_content(
                prompt,
                stream=True,
                generation_config=generation_config,
                **kwargs
            )

            function_call_counter = 0
            for chunk in response:
                if chunk.candidates and chunk.candidates[0].function_calls:
                    function_call = chunk.candidates[0].function_calls[0]
                    function_call_counter += 1

                    if (self.function_call_limit_per_chat is not None and 
                        function_call_counter > self.function_call_limit_per_chat):
                        raise TooManyFunctionCallsException(
                            f"Exceeded function call limit: {self.function_call_limit_per_chat}",
                            self.chat.history
                        )

                    yield {
                        "content": "",
                        "is_complete": False,
                        "function_call": {
                            "name": function_call.name,
                            "args": function_call.args
                        }
                    }

                    api_response = self._call_function(function_call)

                    continuation = self._model.generate_content(
                        Part.from_function_response(
                            name=function_call.name,
                            response={"content": str(api_response)},
                        ),
                        generation_config=generation_config,
                        stream=True
                    )

                    for cont_chunk in continuation:
                        if cont_chunk.text:
                            yield {
                                "content": cont_chunk.text,
                                "is_complete": False,
                                "function_call": None
                            }
                    break

                if chunk.text:
                    yield {
                        "content": chunk.text,
                        "is_complete": False,
                        "function_call": None
                    }

            yield {
                "content": "",
                "is_complete": True,
                "function_call": None
            }

        except Exception as e:
            logging.error(f"Streaming error: {str(e)}")
            yield {
                "content": f"[AGENT ERROR] {str(e)}",
                "is_complete": True,
                "error": True
            }

    @retry(stop=stop_after_attempt(5), 
           wait=wait_exponential(multiplier=2, min=2, max=16), 
           after=log_retry_error, 
           retry=retry_if_not_exception_type(TooManyFunctionCallsException))
    def send_message(
        self,
        msg: str,
        *,
        generation_config: GenerationConfig = None,
        history: List[Dict] = None,
        stream: bool = False
    ) -> Union[Tuple[str, List[Dict]], Generator[Dict[str, Any], None, None]]:
        """Unified interface supporting both streaming and non-streaming."""
        if stream:
            return self.stream_response(msg, generation_config=generation_config, history=history)
        
        initial_history_len = 0
        if history:
            self.set_history(history)
            initial_history_len = len(history)

        response = self.chat.send_message(msg, generation_config=generation_config)
        function_call_counter = 0

        while response.candidates[0].function_calls:
            function_call_counter += 1
            if (self.function_call_limit_per_chat is not None and 
                function_call_counter > self.function_call_limit_per_chat):
                raise TooManyFunctionCallsException(
                    f"Exceeded function call limit: {self.function_call_limit_per_chat}",
                    self.chat.history
                )

            function_call = response.candidates[0].function_calls[0]
            api_response = self._call_function(function_call)

            response = self.chat.send_message(
                Part.from_function_response(
                    name=function_call.name,
                    response={"content": str(api_response)},
                ),
                generation_config=generation_config
            )

        if self.on_message is not None:
            self.on_message(response.text)

        try:
            response_text = response.candidates[0].text
        except Exception as e:
            response_text = f"Error: {str(e)}"

        return response_text, self.chat.history[initial_history_len:]

    def get_history(self) -> List[Dict]:
        """Returns a copy of the chat history."""
        return self.history.copy()
    
    def set_history(self, history: List[Dict]) -> None:
        """Sets the chat history."""
        new_history = []
        for message in history:
            new_history.append(message)
            new_parts = []
            for part in message["raw"].parts:
                if hasattr(part, "text"):
                    new_parts.append(part)
                if hasattr(part, "function_call"):
                    func_name = part.function_call.name
                    if func_name in self.functions:
                        new_parts.append(part)
                if hasattr(part, "function_response"):
                    func_name = part.function_response.name
                    if func_name in self.functions:
                        new_parts.append(part)
            message["raw"]._parts = new_parts
        raw_history = [message["raw"] for message in history]
        self.chat._history = raw_history
        self.history = new_history

    def _call_function(self, function_call) -> Dict:
        """Executes a function call and returns the result."""
        if self.debug:
            print("call function initiated")
            print(str(function_call))
        func = self.functions.get(function_call.name)
        if func:
            try:
                return func(**function_call.args)
            except TypeError as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}"}
            except ValueError as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Value error during function call {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Value error during function call {function_call.name}: {e}\n{stack_trace}"}
            except Exception as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}"}
        return {"error": "Function not found"}

    def _recreate_client(self) -> None:
        """Recreates the chat client."""
        self.chat = self._model.start_chat()

    def _updated_tokens_count(self, updates_tokens_count: Dict, response) -> None:
        """Updates the token count dictionary."""
        current_count = updates_tokens_count.get(response._raw_response.model_version, 0)
        current_count += response.usage_metadata.total_token_count
        updates_tokens_count[response._raw_response.model_version] = current_count

def _generate_function_declaration(func, *, user_set_name=None, user_set_description=None):
    """Generate a FunctionDeclaration from a Python function."""
    func_name = user_set_name or func.__name__
    func_doc = user_set_description or (func.__doc__ or "")

    sig = inspect.signature(func)
    params = {
        "type": "object",
        "properties": {}
    }

    param_descriptions = {}
    if func_doc:
        param_pattern = re.compile(r":param\s+(\w+):\s*(.*)")
        for match in param_pattern.finditer(func_doc):
            param_name, description = match.groups()
            param_descriptions[param_name] = description.strip()

    for name, param in sig.parameters.items():
        type_mapping = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean"
        }
        param_type = type_mapping.get(param.annotation, "string")

        params["properties"][name] = {
            "type": param_type,
            "description": param_descriptions.get(name, "No description provided")
        }

    return FunctionDeclaration(
        name=func_name,
        description=func_doc,
        parameters=params
    )

def create_agent_from_functions_list(
        *,
        model_name: str = DEFAULT_MODEL,
        functions: List[callable] = None,
        function_call_limit_per_chat: int = None,
        system_instruction: str = None,
        debug: bool = False,
        add_scheduling_functions: bool = False,
        gcs_bucket: str = None,
        gcs_blob: str = None,
        delegation_function_prompt: str = None,
        delegates: List[GeminiAgent] = None,
        on_message: callable = None,
        generation_config: GenerationConfig = None
) -> GeminiAgent:
    """Create an agent with custom functions and other parameters."""
    if functions is None:
        functions = []
    if (not gcs_bucket and gcs_blob) or (gcs_bucket and not gcs_blob):
        raise ValueError("Both gcs_bucket and gcs_blob must be provided")

    if add_scheduling_functions:
        scheduler_instance = scheduler.ScheduledTaskExecutor(
            debug=debug, 
            gcs_bucket=gcs_bucket, 
            gcs_blob=gcs_blob
        )
        functions.extend([
            scheduler_instance.add_task,
            scheduler_instance.get_all_jobs,
            scheduler_instance.delete_job,
        ])

    if add_scheduling_functions and not on_message:
        logging.warning("on_message is not set, you may not see the results of the scheduled tasks")

    if functions:
        system_instruction = system_instruction or ""
        system_instruction = f"""{system_instruction}\n\n# IMPORTANT
            You are an agent with the ability to call a set of actions (functions). Follow these rules strictly:

            * You must only call one method at a time.
            * You cannot evaluate expressions as input arguments.
            * You are not allowed to execute arbitrary Python code.
            
            Ensure that you strictly follow these instructions."""

    agent = GeminiAgent(
        delegation_function_prompt=delegation_function_prompt,
        delegates=delegates,
        on_message=on_message,
        functions=functions,
        function_call_limit_per_chat=function_call_limit_per_chat,
        model_name=model_name,
        system_instruction=system_instruction,
        debug=debug,
        generation_config=generation_config
    )

    if add_scheduling_functions:
        scheduler_instance.set_gemini_agent(agent)
        scheduler_instance.start_scheduler()

    return agent
=======
            agent,
            *,
            function_call_limit_per_chat=None,
            on_message=None,
            session_service=None,
            app_name="adk_service",
            events_per_session=-1
    ):
        logging.info("ADKAgentService initializing...")
        self.agent = agent
        self.function_call_limit_per_chat = function_call_limit_per_chat
        self.on_message = on_message
        self.session_service = session_service if session_service else InMemorySessionService()
        self.runners = {}
        self.app_name = app_name
        self.runner_lock = threading.Lock()
        self.events_per_session = events_per_session
        logging.info(f"ADKAgentService initialized with: app_name='{self.app_name}', "
                     f"function_call_limit_per_chat={self.function_call_limit_per_chat}, "
                     f"events_per_session={self.events_per_session}")

    def _maybe_create_chat_session(self, *, user_id, session_id, num_recent_events, events=[]):
        logging.debug(f"Attempting to get/create chat session for user_id='{user_id}', session_id='{session_id}', app_name='{self.app_name}'")
        get_config = None
        if num_recent_events > 0:
            get_config = GetSessionConfig(num_recent_events=num_recent_events)
            logging.debug(f"GetSessionConfig created with num_recent_events={num_recent_events} for session_id='{session_id}'")

        chat_session = self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            config=get_config)

        if not chat_session:
            logging.info(f"No existing session found for session_id='{session_id}'. Creating new session for user_id='{user_id}', app_name='{self.app_name}'.")
            chat_session = self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id)
            logging.info(f"New session created for session_id='{session_id}', user_id='{user_id}', app_name='{self.app_name}'.")
        else:
            logging.info(f"Fetched existing session for session_id='{session_id}', user_id='{user_id}', app_name='{self.app_name}'.")

        if events:
            logging.debug(f"Appending {len(events)} events to session_id='{session_id}'.")
            for event_count, event in enumerate(events, 1):
                self.session_service.append_event(session=chat_session, event=event)
            logging.info(f"Successfully appended {len(events)} events to session_id='{session_id}'.")
        return chat_session

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=16), after=log_retry_error, retry=retry_if_not_exception_type(TooManyFunctionCallsException))
    def send_message(self, msg: str, *, user_id="default_user", session_id=None, events=[]) -> tuple[str, list]:
        """Initiate communication with LLM to execute user's instructions"""
        effective_session_id = session_id if session_id else "new_session"
        logging.info(f"send_message called for user_id='{user_id}', session_id='{effective_session_id}'. Message: '{msg[:100]}{'...' if len(msg) > 100 else ''}'")

        if not session_id:
            session_id = str(uuid.uuid4())
            logging.info(f"No session_id provided. Generated new session_id='{session_id}' for user_id='{user_id}'.")
        
        session = self._maybe_create_chat_session(user_id=user_id, session_id=session_id, num_recent_events=self.events_per_session, events=events)
        
        logging.debug(f"about to send msg: {msg}")
        
        runner_id = user_id + session_id
        logging.debug(f"Runner ID for session_id='{session_id}' is '{runner_id}'.")
        runner_instance = self.runners.get(runner_id)
        if not runner_instance:
            with self.runner_lock:
                runner_instance = self.runners.get(runner_id) 
                if not runner_instance:
                    try:
                        logging.info(f"Creating new Runner instance for runner_id='{runner_id}', session_id='{session_id}'.")
                        runner_instance = Runner(
                            agent=self.agent,
                            app_name=self.app_name,
                            session_service=self.session_service
                        )
                        self.runners[runner_id] = runner_instance
                        logging.info(f"Runner instance created and cached for runner_id='{runner_id}', session_id='{session_id}'.")
                    except Exception as e:
                        logging.exception(f"Fatal Error creating agent/runner for session_id='{session_id}', runner_id='{runner_id}': {e}")
                        
        
        if not runner_instance:
            logging.critical(f"Runner instance is None for session_id='{session_id}', runner_id='{runner_id}'. Cannot proceed.")
            
            return "Failed to initialize agent runner.", []

        user_content = genai_types.Content(role='user', parts=[genai_types.Part(text=msg)])
        final_response_text = ""
        function_call_counter = 0
        
        logging.info(f"Preparing to call runner_instance.run() for session_id='{session_id}'.")
        try:
            for event in runner_instance.run( 
                user_id=user_id,
                new_message=user_content,
                session_id=session_id):
                function_call_counter = function_call_counter + 1
                if self.function_call_limit_per_chat is not None and function_call_counter >= self.function_call_limit_per_chat:
                    raise TooManyFunctionCallsException(
                        f"Exceed allowed number of function calls: {self.function_call_limit_per_chat}"
                    )
                

                logging.debug(f"ADK Event ({session_id}): Author={event.author}, Content={event.content}")

                if event.error_message:
                    logging.error(f"ADK Runner Error ({session_id}): {event.error_message}")
                    break
                logging.debug(str(event))

                if event.is_final_response() and event.content and event.content.parts:
                    text_part = next((part.text for part in event.content.parts if part.text), None)
                    if text_part:
                        final_response_text = text_part
                        logging.debug(f"ADK signaled final response with text ({session_id}): {final_response_text}")
                    break
        except genai_types.BlockedPromptException as e:
                logging.warning(f"Prompt was blocked for session {session_id}: {e}")
                final_response_text = "Your prompt was blocked. Please modify your prompt and try again."
        except genai_types.StopCandidateException as e:
                logging.warning(f"Content generation stopped for session {session_id}: {e}")
                final_response_text = "The response could not be completed. Please try again."
        except google_exceptions.DeadlineExceeded as e:
                logging.error(f"API request timed out during runner.run for session_id='{session_id}': {e}")
                final_response_text = "The request timed out. Please try again later."
        except google_exceptions.GoogleAPIError as e:
                logging.error(f"A Google API error occurred during runner.run for session_id='{session_id}': {e}")
                final_response_text = "An API error occurred. Please try again later."
        except Exception as e: 
                logging.exception(f"An unexpected error occurred during runner.run for session_id='{session_id}': {e}")
                
                final_response_text = "An unexpected error occurred. Please try again."
        
        logging.info(f"runner_instance.run() completed or errored for session_id='{session_id}'.")

        
        if final_response_text:
                logging.debug(f"Final response text for session_id='{session_id}' (prior to on_message): '{final_response_text[:100]}{'...' if len(final_response_text) > 100 else ''}'")
        else:
                logging.debug(f"No final response text was set from runner.run() for session_id='{session_id}'.")
        
        if self.on_message:
            logging.info(f"Calling on_message callback for session_id='{session_id}'.")
            self.on_message(final_response_text)
            logging.info(f"on_message callback completed for session_id='{session_id}'.")
        
        logging.info(f"Returning final response for session_id='{session_id}'. Response: '{final_response_text[:100]}{'...' if len(final_response_text) > 100 else ''}'")
        
        returned_events = self._maybe_create_chat_session(
            session_id=session_id, user_id=user_id, num_recent_events=self.events_per_session).events
        logging.debug(f"Returning {len(returned_events)} events for session_id='{session_id}'.")
        return final_response_text, returned_events
