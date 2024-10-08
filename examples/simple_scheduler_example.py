"""This example illustrates a periodical task execution using a scheduler"""

import time
import vertexai
from config import (PROJECT_ID, REGION, SIMPLE_MODEL)
from gemini_agents_toolkit import agent


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


def print_msg_from_agent(msg: str):
    """print message in console"""
    print(msg)


vertexai.init(project=PROJECT_ID, location=REGION)

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions,
                                                          model_name=SIMPLE_MODEL,
                                                          add_scheduling_functions=True,
                                                          on_message=print_msg_from_agent)

# no need to print result directly since we passed to agent on_message
duck_comms_agent.send_message("can you be saying, each minute, to the duck that I am hungry")

# wait 3 min to see results
time.sleep(180)
