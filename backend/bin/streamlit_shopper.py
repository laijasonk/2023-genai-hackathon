import os
import sys
import json

import streamlit
from streamlit_chat import message

# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("./"))
from src.models import vertexai_shopper


##################################################
#
# Configuration
#
##################################################

# Load the config
with open("./data/configs/vertexai_shopper.json", "r") as fn:
    config = json.load(fn)

# Config streamlit
streamlit.set_page_config(
    page_title="Personal Shopper",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# Initialize the chatbot
chatbot = vertexai_shopper.VertexAIShopper(config)


##################################################
#
# Helper Functions
#
##################################################


def generate_response(user_input):
    chatbot.add_streamlit_history(
        streamlit.session_state["past"], streamlit.session_state["generated"]
    )
    response = chatbot.add_user_input(user_input)
    print(response)
    return response


def get_text():
    # input_text = streamlit.text_input("You: ", "Hello, how are you?", key="input")
    input_text = streamlit.text_input("You: ", key="input", on_change=on_submit_text)
    return streamlit.session_state["prompt"]


def on_submit_text():
    streamlit.session_state["prompt"] = streamlit.session_state["input"]
    streamlit.session_state["input"] = ""
    return None


def set_recommendations(response):
    _validate_recommendation(response, "outerwear")
    _validate_recommendation(response, "top")
    _validate_recommendation(response, "bottom")
    if "dress" in streamlit.session_state["top1"]:
        streamlit.session_state["bottom1"] = streamlit.session_state["top1"]
    if "dress" in streamlit.session_state["top2"]:
        streamlit.session_state["bottom2"] = streamlit.session_state["top2"]
    if "dress" in streamlit.session_state["top3"]:
        streamlit.session_state["bottom3"] = streamlit.session_state["top3"]
    if "dress" in streamlit.session_state["bottom1"]:
        streamlit.session_state["top1"] = streamlit.session_state["bottom1"]
    if "dress" in streamlit.session_state["bottom2"]:
        streamlit.session_state["top2"] = streamlit.session_state["bottom2"]
    if "dress" in streamlit.session_state["bottom3"]:
        streamlit.session_state["top3"] = streamlit.session_state["bottom3"]
    return None


def _validate_recommendation(response, clothing):
    try:
        if response[clothing]:
            if len(response[clothing]) < 2:
                response[clothing].append(response[clothing][0])
        else:
            response[clothing] = ["", ""]

        if response[clothing + "_color"]:
            if len(response[clothing + "_color"]) < 2:
                response[clothing + "_color"].append(response[clothing + "_color"][0])
        else:
            response[clothing + "_color"] = ["", ""]

        streamlit.session_state[
            clothing + "1"
        ] = f"{response[clothing + '_color'][0]} {response[clothing][0]}"
        streamlit.session_state[
            clothing + "2"
        ] = f"{response[clothing + '_color'][0]} {response[clothing][1]}"
        streamlit.session_state[
            clothing + "3"
        ] = f"{response[clothing + '_color'][1]} {response[clothing][0]}"
    except:
        streamlit.session_state[clothing + "1"] = ""
        streamlit.session_state[clothing + "2"] = ""
        streamlit.session_state[clothing + "3"] = ""

    return None


def set_demographics(response):
    _validate_demographics(response, "gender")
    _validate_demographics(response, "age")
    _validate_demographics(response, "income")
    _validate_demographics(response, "style")
    return None


def _validate_demographics(response, demographics):
    try:
        streamlit.session_state[demographics] = response["customer_" + demographics]
    except:
        streamlit.session_state[demographics] = ""

    return None


##################################################
#
# Persistent/Session Variables
#
##################################################

# Chat history
if "past" not in streamlit.session_state:
    streamlit.session_state["past"] = []
if "generated" not in streamlit.session_state:
    streamlit.session_state["generated"] = []

# Current input prompt
if "prompt" not in streamlit.session_state:
    streamlit.session_state["prompt"] = ""

# Current clothing recommendation
if "outerwear1" not in streamlit.session_state:
    streamlit.session_state["outerwear1"] = ""
if "outerwear2" not in streamlit.session_state:
    streamlit.session_state["outerwear2"] = ""
if "outerwear3" not in streamlit.session_state:
    streamlit.session_state["outerwear3"] = ""
if "top1" not in streamlit.session_state:
    streamlit.session_state["top1"] = ""
if "top2" not in streamlit.session_state:
    streamlit.session_state["top2"] = ""
if "top3" not in streamlit.session_state:
    streamlit.session_state["top3"] = ""
if "bottom1" not in streamlit.session_state:
    streamlit.session_state["bottom1"] = ""
if "bottom2" not in streamlit.session_state:
    streamlit.session_state["bottom2"] = ""
if "bottom3" not in streamlit.session_state:
    streamlit.session_state["bottom3"] = ""

# Customer insights
if "gender" not in streamlit.session_state:
    streamlit.session_state["gender"] = ""
if "age" not in streamlit.session_state:
    streamlit.session_state["age"] = ""
if "income" not in streamlit.session_state:
    streamlit.session_state["income"] = ""
if "style" not in streamlit.session_state:
    streamlit.session_state["style"] = ""


##################################################
#
# Streamlit App: Main Display
#
##################################################

# Title text
streamlit.title("Personal Clothing Shopper")

# Recommendation
clothing1, clothing2, clothing3 = streamlit.columns(3)
recommendation1 = clothing1.text(f"Outer:\nTop:\nBottom:")
recommendation2 = clothing2.text(f"Outer:\nTop:\nBottom:")
recommendation3 = clothing3.text(f"Outer:\nTop:\nBottom:")

# Sidebar (hidden by default)
streamlit.sidebar.title("Customer Insights")
demographics1 = streamlit.sidebar.text(f"Guessed Gender:")
demographics2 = streamlit.sidebar.text(f"Guessed Age:")
demographics3 = streamlit.sidebar.text(f"Guessed Income:")
demographics4 = streamlit.sidebar.text(f"Guessed Style:")


##################################################
#
# Streamlit App: Chat Display
#
##################################################

# User input
user_input = get_text()

# Handle input data
if user_input:
    # Get response from PaLM
    response = generate_response(user_input)

    # Add user input and response to session
    streamlit.session_state.past.append(user_input)
    streamlit.session_state.generated.append(response["response"])

    # Determine what clothing to suggest
    set_recommendations(response)
    recommendation1.text(
        f"Outer: {streamlit.session_state['outerwear1']}\nTop: {streamlit.session_state['top1']}\nBottom: {streamlit.session_state['bottom1']}"
    )
    recommendation2.text(
        f"Outer: {streamlit.session_state['outerwear2']}\nTop: {streamlit.session_state['top2']}\nBottom: {streamlit.session_state['bottom2']}"
    )
    recommendation3.text(
        f"Outer: {streamlit.session_state['outerwear3']}\nTop: {streamlit.session_state['top3']}\nBottom: {streamlit.session_state['bottom3']}"
    )

    # Guess some customer insights
    set_demographics(response)
    demographics1.text(f"Guessed Gender: {streamlit.session_state['gender']}")
    demographics2.text(f"Guessed Age: {streamlit.session_state['age']}")
    demographics3.text(f"Guessed Income: {streamlit.session_state['income']}")
    demographics4.text(f"Guessed Style: {streamlit.session_state['style']}")

else:
    streamlit.session_state.past.append("Hi")
    streamlit.session_state.generated.append(
        "I am your personal clothing shopper! How can I help you?"
    )

# Display the chatbot message history
if streamlit.session_state["generated"]:
    for i in range(len(streamlit.session_state["generated"]) - 1, -1, -1):
        message(
            streamlit.session_state["generated"][i],
            avatar_style="avataaars",
            seed=1234,
            key=str(i),
        )
        if not i == 0:
            message(
                streamlit.session_state["past"][i],
                avatar_style="icons",
                seed=1234,
                is_user=True,
                key=str(i) + "_user",
            )
