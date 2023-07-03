"""Streamlit app for a personal shopper.

Requires gcloud authentication to access the PaLM API. Run prototype UI
with `streamlit run ./bin/streamlit_shopper.py`
"""

import os
import sys
import json

import streamlit
from streamlit_chat import message

import numpy
from PIL import Image

# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("./"))
from src.models import vertexai_shopper
from src.models import vertexai_styleguide
from src.utils import shopper_utils as utils


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

def init_session_state(variable, value):
    if str(variable) not in streamlit.session_state:
        streamlit.session_state[str(variable)] = value
    return None


def get_text_button():
    input_text = streamlit.text_input("You: ", key="input", on_change=on_submit_text)
    return streamlit.session_state["prompt"]


def on_submit_text():
    streamlit.session_state["prompt"] = streamlit.session_state["input"]
    streamlit.session_state["input"] = ""
    return None


def generate_style_guide():
    # Get a response and accompanying model
    response, model = utils.generate_style_guide_response(streamlit.session_state)

    # Check if correctly generated
    if not response == {}:
        # Create powerpoint
        model.generate_powerpoint(response, "./style_guide.pptx")

        # Allow downloading
        with open("style_guide.pptx", "rb") as file:
            download_style_button = style2.download_button(
                label="Download Style Guide",
                data=file,
                file_name="style_guide.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
    else:
        style2.text("Style guide\nfailed to\ngenerate")

    return None


##################################################
#
# Persistent/Session Variables
#
##################################################

# Chat history
init_session_state("past", [])
init_session_state("generated", [])
init_session_state("prompt", "")

# Current clothing recommendation
init_session_state("outerwear1", "")
init_session_state("outerwear2", "")
init_session_state("outerwear3", "")
init_session_state("top1", "")
init_session_state("top2", "")
init_session_state("top3", "")
init_session_state("bottom1", "")
init_session_state("bottom2", "")
init_session_state("bottom3", "")

# Current color recommendation
init_session_state("outerwear_color1", "")
init_session_state("outerwear_color2", "")
init_session_state("outerwear_color3", "")
init_session_state("top_color1", "")
init_session_state("top_color2", "")
init_session_state("top_color3", "")
init_session_state("bottom_color1", "")
init_session_state("bottom_color2", "")
init_session_state("bottom_color3", "")

# Customer insights
init_session_state("gender", "")
init_session_state("age", "")
init_session_state("income", "")
init_session_state("style", "")


##################################################
#
# Streamlit App: Main Display
#
##################################################

# # Title text
# streamlit.title("Personal Clothing Shopper")

# Base images
man_base = Image.open("data/templates/man_base.png")
woman_base = Image.open("data/templates/woman_base.png")
unknown_image = Image.open("data/templates/man_unknown.png")

# Recommendation
clothing1, clothing2, clothing3 = streamlit.columns(3)
image1 = clothing1.image(unknown_image)
image2 = clothing2.image(unknown_image)
image3 = clothing3.image(unknown_image)
recommendation1 = clothing1.text(f"Outer:\nTop:\nBottom:")
recommendation2 = clothing2.text(f"Outer:\nTop:\nBottom:")
recommendation3 = clothing3.text(f"Outer:\nTop:\nBottom:")
streamlit.markdown("""---""")

# Sidebar button states
generate_style_button = False
download_style_button = False

# Sidebar (hidden by default)
streamlit.sidebar.title("Customer Insights")
demographics1 = streamlit.sidebar.text(f"Guessed Gender:")
demographics2 = streamlit.sidebar.text(f"Guessed Age:")
demographics3 = streamlit.sidebar.text(f"Guessed Income:")
demographics4 = streamlit.sidebar.text(f"Guessed Style:")
streamlit.sidebar.markdown("""---""")
style1, style2 = streamlit.sidebar.columns(2)
generate_style_button = style1.button("Generate Style Guide")
if generate_style_button:
    download_style_button = generate_style_guide()

##################################################
#
# Streamlit App: Chat Display
#
##################################################

# User input
user_input = get_text_button()

# Handle input data
if user_input:
    # Get response from PaLM
    response = utils.generate_response(
        chatbot, 
        user_input, 
        streamlit.session_state["past"], 
        streamlit.session_state["generated"]
    )

    # Add user input and response to session
    if streamlit.session_state["past"] and streamlit.session_state["generated"]:
        streamlit.session_state["past"].append(user_input)
        streamlit.session_state["generated"].append(response["response"])

    # Determine what clothing to suggest
    streamlit.session_state = utils.set_recommendations(streamlit.session_state, response)

    # Change the image according to the recommendation
    streamlit.session_state, image1, image2, image3 = utils.change_image(streamlit.session_state, man_base, woman_base, image1, image2, image3)

    # Special case for dresses and skirts
    if response["exception"]:
        image1.image = unknown_image
        image2.image = unknown_image
        image3.image = unknown_image
        recommendation1.text = f"Outer:\nTop:\nBottom:"
        recommendation2.text = f"Outer:\nTop:\nBottom:"
        recommendation3.text = f"Outer:\nTop:\nBottom:"
    else:
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
    streamlit.session_state = utils.set_demographics(streamlit.session_state, response)
    demographics1.text(f"Guessed Gender: {streamlit.session_state['gender']}")
    demographics2.text(f"Guessed Age: {streamlit.session_state['age']}")
    demographics3.text(f"Guessed Income: {streamlit.session_state['income']}")
    demographics4.text(f"Guessed Style: {streamlit.session_state['style']}")

elif generate_style_button:
    pass

elif download_style_button:
    pass

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

