import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from json_agent import create_agent as create_json_agent
from summarizer_agent import create_agent as create_summarizer_agent

yaml_path = os.path.join(os.path.dirname(__file__), "test.yaml")
if not os.path.exists(yaml_path):
    raise FileNotFoundError("test.yaml not found!")

with open(yaml_path, "r", encoding="utf-8") as f:
    yaml_content = f.read()

json_agent = create_json_agent()
summarizer_agent = create_summarizer_agent()

json_output, _ = json_agent.send_message(yaml_content)
print("\n--- JSON Output ---\n")
print(json_output)

summary, _ = summarizer_agent.send_message(json_output)
print("\n--- Summary ---\n")
print(summary)
