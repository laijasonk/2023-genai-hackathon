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
        streamlit.session_state["past"],
        streamlit.session_state["generated"],
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
        ] = f"{response[clothing + '_color'][1]} {response[clothing][0]}"
        streamlit.session_state[
            clothing + "3"
        ] = f"{response[clothing + '_color'][0]} {response[clothing][1]}"

        streamlit.session_state[clothing + "_color1"] = response[clothing + "_color"][0]
        streamlit.session_state[clothing + "_color2"] = response[clothing + "_color"][1]
        streamlit.session_state[clothing + "_color3"] = response[clothing + "_color"][0]
    except:
        streamlit.session_state[clothing + "1"] = ""
        streamlit.session_state[clothing + "2"] = ""
        streamlit.session_state[clothing + "3"] = ""
        streamlit.session_state[clothing + "_color1"] = ""
        streamlit.session_state[clothing + "_color2"] = ""
        streamlit.session_state[clothing + "_color3"] = ""

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


def generate_style_guide():
    # Load the config
    with open("./data/configs/vertexai_styleguide.json", "r") as fn:
        config = json.load(fn)

    # Initialize the model
    model = vertexai_styleguide.VertexAIStyleGuide(config)

    # Pass customer insights into model
    model.customer_insights = {
        "gender": streamlit.session_state["gender"],
        "age": streamlit.session_state["age"],
        "income": streamlit.session_state["income"],
        "style": streamlit.session_state["style"],
        "outerwear1": streamlit.session_state["outerwear1"],
        "outerwear2": streamlit.session_state["outerwear2"],
        "outerwear3": streamlit.session_state["outerwear3"],
        "top1": streamlit.session_state["top1"],
        "top2": streamlit.session_state["top2"],
        "top3": streamlit.session_state["top3"],
        "bottom1": streamlit.session_state["bottom1"],
        "bottom2": streamlit.session_state["bottom2"],
        "bottom3": streamlit.session_state["bottom3"],
    }

    # Setup model
    model.use_default_template()
    model.setup_model()

    # Get response
    response = model.add_user_input(model.customer_insights)

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


def change_image_colors(base_image, image, outerwear_color, top_color, bottom_color):
    data = numpy.array(base_image)

    if not outerwear_color == "":
        # Outerwear color
        r1, g1, b1 = 17, 17, 17  # Original value
        r2, g2, b2 = _get_rgb(outerwear_color)
        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

    if not top_color == "":
        # Top color
        r1, g1, b1 = 34, 34, 34  # Original value
        r2, g2, b2 = _get_rgb(top_color)
        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

    if not bottom_color == "":
        # Bottom color
        r1, g1, b1 = 51, 51, 51  # Original value
        r2, g2, b2 = _get_rgb(bottom_color)
        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

    # Replace image
    new_image = Image.fromarray(data)
    image.image(new_image)

    return None


def _get_rgb(color):
    colors = {
        "red": [178, 34, 34],
        "green": [60, 179, 113],
        "blue": [100, 149, 237],
        "white": [240, 240, 240],
        "black": [30, 30, 30],
        "grey": [211, 211, 211],
        "beige": [245, 245, 220],
        "brown": [160, 82, 45],
        "pink": [255, 182, 193],
        "orange": [255, 165, 0],
        "purple": [186, 85, 211],
        "yellow": [229, 229, 0],
        "navy": [0, 0, 128],
        "khaki": [240, 230, 140],
    }
    if str(color).lower() in colors:
        r, g, b = colors[color]
    else:
        r, g, b = [211, 211, 211]

    return r, g, b


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
if "outerwear_color1" not in streamlit.session_state:
    streamlit.session_state["outerwear_color1"] = ""
if "outerwear_color2" not in streamlit.session_state:
    streamlit.session_state["outerwear_color2"] = ""
if "outerwear_color3" not in streamlit.session_state:
    streamlit.session_state["outerwear_color3"] = ""
if "top_color1" not in streamlit.session_state:
    streamlit.session_state["top_color1"] = ""
if "top_color2" not in streamlit.session_state:
    streamlit.session_state["top_color2"] = ""
if "top_color3" not in streamlit.session_state:
    streamlit.session_state["top_color3"] = ""
if "bottom_color1" not in streamlit.session_state:
    streamlit.session_state["bottom_color1"] = ""
if "bottom_color2" not in streamlit.session_state:
    streamlit.session_state["bottom_color2"] = ""
if "bottom_color3" not in streamlit.session_state:
    streamlit.session_state["bottom_color3"] = ""

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

    if streamlit.session_state["bottom1"][-5:] in ["dress"]:
        streamlit.session_state["top1"] = streamlit.session_state["bottom1"]
        streamlit.session_state["top_color1"] = streamlit.session_state["bottom_color1"]
        base_image = woman_base
    elif streamlit.session_state["top1"][-5:] in ["dress", "skirt"]:
        streamlit.session_state["bottom1"] = streamlit.session_state["top1"]
        streamlit.session_state["bottom_color1"] = streamlit.session_state["top_color1"]
        base_image = woman_base
    elif streamlit.session_state["bottom1"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base
    change_image_colors(
        base_image,
        image1,
        streamlit.session_state["outerwear_color1"],
        streamlit.session_state["top_color1"],
        streamlit.session_state["bottom_color1"],
    )

    if streamlit.session_state["bottom2"][-5:] in ["dress"]:
        streamlit.session_state["top2"] = streamlit.session_state["bottom2"]
        streamlit.session_state["top_color2"] = streamlit.session_state["bottom_color2"]
        base_image = woman_base
    elif streamlit.session_state["top2"][-5:] in ["dress", "skirt"]:
        streamlit.session_state["bottom2"] = streamlit.session_state["top2"]
        streamlit.session_state["bottom_color2"] = streamlit.session_state["top_color2"]
        base_image = woman_base
    elif streamlit.session_state["bottom2"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base
    change_image_colors(
        base_image,
        image2,
        streamlit.session_state["outerwear_color2"],
        streamlit.session_state["top_color2"],
        streamlit.session_state["bottom_color2"],
    )

    if streamlit.session_state["bottom3"][-5:] in ["dress"]:
        streamlit.session_state["top3"] = streamlit.session_state["bottom3"]
        streamlit.session_state["top_color3"] = streamlit.session_state["bottom_color3"]
        base_image = woman_base
    elif streamlit.session_state["top3"][-5:] in ["dress", "skirt"]:
        streamlit.session_state["bottom3"] = streamlit.session_state["top3"]
        streamlit.session_state["bottom_color3"] = streamlit.session_state["top_color3"]
        base_image = woman_base
    elif streamlit.session_state["bottom3"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base
    change_image_colors(
        base_image,
        image3,
        streamlit.session_state["outerwear_color3"],
        streamlit.session_state["top_color3"],
        streamlit.session_state["bottom_color3"],
    )

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
    set_demographics(response)
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
