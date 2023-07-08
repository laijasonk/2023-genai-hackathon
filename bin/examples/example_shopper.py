#!/usr/bin/env python3

import os
import sys
import json

# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("./"))
from src.models import vertexai_shopper

# Load the config
with open("./data/configs/vertexai_shopper.json", "r") as fn:
    config = json.load(fn)

# Initialize the chatbot
chatbot = vertexai_shopper.VertexAIShopper(config)

# Introductory text
print("Starting example AI chatbot")
print('Type "quit" or CTRL-C to quit')
print()

# Run an interactive chatbot
user_input = input("Customer: ")
while not user_input == "quit":
    response = chatbot.add_user_input(user_input)
    print(f"{response}".strip())
    print()
    user_input = input("Customer: ")
