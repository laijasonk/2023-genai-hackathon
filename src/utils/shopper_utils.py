"""Helper functions for shopper Streamlit app

Various code to help support the bin/streamlit_shopper.py app.
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
from src.models import vertexai_promovideo

colors = {
    "red": [178, 34, 34],
    "green": [60, 179, 113],
    "blue": [100, 149, 237],
    "white": [240, 240, 240],
    "black": [30, 30, 30],
    "grey": [200, 200, 200],
    "light grey": [230, 230, 230],
    "dark grey": [150, 150, 150],
    "beige": [245, 245, 220],
    "brown": [160, 82, 45],
    "pink": [255, 182, 193],
    "orange": [255, 165, 0],
    "purple": [186, 85, 211],
    "yellow": [229, 229, 0],
    "navy": [0, 0, 128],
    "navy blue": [0, 0, 150],
    "khaki": [240, 230, 140],
    "light blue": [173, 216, 230],
    "baby blue": [137, 207, 240],
}


def generate_response(chatbot, user_input, past, generated):
    """Return a response from the LLM.

    Args:
        chatbot (obj): Chatbot object
        user_input (str): Input prompt to LLM
    Returns:
        response (dict): Response from LLM
    """

    chatbot.add_streamlit_history(past, generated)
    response = chatbot.add_user_input(user_input)

    print(response)
    return response


def set_recommendations(session_state, response):
    """Change the session_state with latest response.

    Args:
        session_state (dict): Streamlit session state
        response (dict): LLM response
    Returns:
        session_state (dict): Modified Streamlit session state
    """

    session_state = _validate_recommendation(session_state, response, "outerwear")
    session_state = _validate_recommendation(session_state, response, "top")
    session_state = _validate_recommendation(session_state, response, "bottom")

    return session_state


def _validate_recommendation(session_state, response, clothing):
    """Private method to validate input recommendation.

    Args:
        session_state (dict): Streamlit session state
        response (dict): LLM response
        clothing (string): Type of clothing (e.g. outerwear, top, bottom)
    Returns:
        session_state (dict): Modified Streamlit session state
    """

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

        response[clothing][0] = _remove_color_words(response[clothing][0])
        response[clothing][1] = _remove_color_words(response[clothing][1])

        session_state[
            clothing + "1"
        ] = f"{response[clothing + '_color'][0]} {response[clothing][0]}"
        session_state[
            clothing + "2"
        ] = f"{response[clothing + '_color'][1]} {response[clothing][0]}"
        session_state[
            clothing + "3"
        ] = f"{response[clothing + '_color'][0]} {response[clothing][1]}"

        session_state[clothing + "_color1"] = response[clothing + "_color"][0]
        session_state[clothing + "_color2"] = response[clothing + "_color"][1]
        session_state[clothing + "_color3"] = response[clothing + "_color"][0]
    except:
        pass
        # session_state[clothing + "1"] = ""
        # session_state[clothing + "2"] = ""
        # session_state[clothing + "3"] = ""
        # session_state[clothing + "_color1"] = ""
        # session_state[clothing + "_color2"] = ""
        # session_state[clothing + "_color3"] = ""

    return session_state


def _remove_color_words(text):
    """Private function to remove colors from text.

    Args:
        text (str): Haystack to remove colors from
    Returns:
        new_text (str): Haystack with needles removed
    """

    new_text = " ".join(
        word for word in text.split() if word not in list(colors.keys())
    )
    return new_text.strip()


def set_demographics(session_state, response):
    """Change the session_state with latest demographics.

    Args:
        session_state (dict): Streamlit session state
        response (dict): LLM response
    Returns:
        session_state (dict): Modified Streamlit session state
    """

    session_state = _validate_demographics(session_state, response, "gender")
    session_state = _validate_demographics(session_state, response, "age")
    session_state = _validate_demographics(session_state, response, "income")
    session_state = _validate_demographics(session_state, response, "style")

    return session_state


def _validate_demographics(session_state, response, demographics):
    """Private method to validate input demographics.

    Args:
        session_state (dict): Streamlit session state
        response (dict): LLM response
        demographics (string): Type of info (e.g. gender, age, income, style)
    Returns:
        session_state (dict): Modified Streamlit session state
    """

    try:
        session_state[demographics] = response["customer_" + demographics]
    except:
        session_state[demographics] = ""

    return session_state


def change_image(session_state, man_base, woman_base, image1, image2, image3):
    """Modify the images according to response.

    Args:
        session_state (dict): Streamlit session state
        man_base (numpy): Base image of a man
        woman_base (numpy): Base image of a woman
        image1 (obj): Streamlit image recommendation 1
        image2 (obj): Streamlit image recommendation 1
        image3 (obj): Streamlit image recommendation 1
    Returns:
        session_state (dict): Modified Streamlit session state
        image1 (obj): Modified Streamlit image recommendation 1
        image2 (obj): Modified Streamlit image recommendation 1
        image3 (obj): Modified Streamlit image recommendation 1
    """

    # Recommendation 1
    if session_state["bottom1"][-5:] in ["dress"]:
        session_state["top1"] = session_state["bottom1"]
        session_state["top_color1"] = session_state["bottom_color1"]
        base_image = woman_base
    elif session_state["top1"][-5:] in ["dress", "skirt"]:
        session_state["bottom1"] = session_state["top1"]
        session_state["bottom_color1"] = session_state["top_color1"]
        base_image = woman_base
    elif session_state["bottom1"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base

    image1 = _change_colors(
        base_image,
        image1,
        session_state["outerwear_color1"],
        session_state["top_color1"],
        session_state["bottom_color1"],
        "./data/templates/clothing1.png",
    )

    # Recommendation 2
    if session_state["bottom2"][-5:] in ["dress"]:
        session_state["top2"] = session_state["bottom2"]
        session_state["top_color2"] = session_state["bottom_color2"]
        base_image = woman_base
    elif session_state["top2"][-5:] in ["dress", "skirt"]:
        session_state["bottom2"] = session_state["top2"]
        session_state["bottom_color2"] = session_state["top_color2"]
        base_image = woman_base
    elif session_state["bottom2"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base

    image2 = _change_colors(
        base_image,
        image2,
        session_state["outerwear_color2"],
        session_state["top_color2"],
        session_state["bottom_color2"],
        "./data/templates/clothing2.png",
    )

    # Recommendation 3
    if session_state["bottom3"][-5:] in ["dress"]:
        session_state["top3"] = session_state["bottom3"]
        session_state["top_color3"] = session_state["bottom_color3"]
        base_image = woman_base
    elif session_state["top3"][-5:] in ["dress", "skirt"]:
        session_state["bottom3"] = session_state["top3"]
        session_state["bottom_color3"] = session_state["top_color3"]
        base_image = woman_base
    elif session_state["bottom3"][-5:] in ["skirt"]:
        base_image = woman_base
    else:
        base_image = man_base

    image3 = _change_colors(
        base_image,
        image3,
        session_state["outerwear_color3"],
        session_state["top_color3"],
        session_state["bottom_color3"],
        "./data/templates/clothing3.png",
    )

    return session_state, image1, image2, image3


def _change_colors(base_image, image, out_color, top_color, bot_color, image_fn):
    """Private function to change the colors of an image.

    Args:
        base_image (numpy): Base image
        image (obj): Streamlit image
        out_color (str): Color of outerwear
        top_color (str): Color of top
        bot_color (str): Color of bottoms
    Returns:
        image (obj): Modified Streamlit image
    """

    # Convert image to numpy if not already
    data = numpy.array(base_image)

    if not out_color == "":
        # Outerwear color
        r1, g1, b1 = 17, 17, 17  # Original value
        r2, g2, b2 = _get_rgb(out_color)
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

    if not bot_color == "":
        # Bottom color
        r1, g1, b1 = 51, 51, 51  # Original value
        r2, g2, b2 = _get_rgb(bot_color)
        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

    # Replace image
    new_image = Image.fromarray(data)
    #new_image.save(image_fn)

    return image.image(new_image)


def _get_rgb(color):
    """Private function to convert color name to rgb.

    Args:
        color (string): Name of color
    Returns:
        r (int): Red value
        g (int): Green value
        b (int): Blue value
    """

    if str(color).lower() in colors:
        r, g, b = colors[color]
    else:
        r, g, b = [211, 211, 211]

    return r, g, b


def change_recommendation_text(
    session_state, recommendation1, recommendation2, recommendation3
):
    """Modify the recommendation text according to response.

    Args:
        session_state (dict): Streamlit session state
        recommendation1 (obj): Streamlit recommendation 1
        recommendation2 (obj): Streamlit recommendation 2
        recommendation3 (obj): Streamlit recommendation 3
    Returns:
        recommendation1 (obj): Modified Streamlit recommendation 1
        recommendation2 (obj): Modified Streamlit recommendation 2
        recommendation3 (obj): Modified Streamlit recommendation 3
    """

    outerwear1 = str(session_state["outerwear1"])
    outerwear1_link = "https://amazon.com/s?k=" + outerwear1.replace(" ", "+")
    outerwear2 = str(session_state["outerwear2"])
    outerwear2_link = "https://amazon.com/s?k=" + outerwear2.replace(" ", "+")
    outerwear3 = str(session_state["outerwear3"])
    outerwear3_link = "https://amazon.com/s?k=" + outerwear3.replace(" ", "+")

    top1 = str(session_state["top1"])
    top1_link = "https://amazon.com/s?k=" + top1.replace(" ", "+")
    top2 = str(session_state["top2"])
    top2_link = "https://amazon.com/s?k=" + top2.replace(" ", "+")
    top3 = str(session_state["top3"])
    top3_link = "https://amazon.com/s?k=" + top3.replace(" ", "+")

    bottom1 = str(session_state["bottom1"])
    bottom1_link = "https://amazon.com/s?k=" + bottom1.replace(" ", "+")
    bottom2 = str(session_state["bottom2"])
    bottom2_link = "https://amazon.com/s?k=" + bottom2.replace(" ", "+")
    bottom3 = str(session_state["bottom3"])
    bottom3_link = "https://amazon.com/s?k=" + bottom3.replace(" ", "+")

    recommendation1.markdown(
        f"Outer: [{outerwear1}]({outerwear1_link})<br>Top: [{top1}]({top1_link})<br>Bottom: [{bottom1}]({bottom1_link})",
        unsafe_allow_html=True,
    )
    recommendation2.markdown(
        f"Outer: [{outerwear2}]({outerwear2_link})<br>Top: [{top2}]({top2_link})<br>Bottom: [{bottom2}]({bottom2_link})",
        unsafe_allow_html=True,
    )
    recommendation3.markdown(
        f"Outer: [{outerwear3}]({outerwear3_link})<br>Top: [{top3}]({top3_link})<br>Bottom: [{bottom3}]({bottom3_link})",
        unsafe_allow_html=True,
    )

    return recommendation1, recommendation2, recommendation3


def generate_style_guide_response(session_state):
    """Return the response from the style guide model.

    Args:
        session_state (dict): Streamlit session state
    Returns:
        response (dict): Response from LLM
        model (obj): Model object
    """

    # Load the config
    with open("./data/configs/vertexai_styleguide.json", "r") as fn:
        config = json.load(fn)

    # Initialize the model
    model = vertexai_styleguide.VertexAIStyleGuide(config)

    # Pass customer insights into model
    model.customer_insights = {
        "gender": session_state["gender"],
        "age": session_state["age"],
        "income": session_state["income"],
        "style": session_state["style"],
        "outerwear1": session_state["outerwear1"],
        "outerwear2": session_state["outerwear2"],
        "outerwear3": session_state["outerwear3"],
        "top1": session_state["top1"],
        "top2": session_state["top2"],
        "top3": session_state["top3"],
        "bottom1": session_state["bottom1"],
        "bottom2": session_state["bottom2"],
        "bottom3": session_state["bottom3"],
    }

    # Setup model
    model.use_default_template()
    model.setup_model()

    # Get response
    response = model.add_user_input(model.customer_insights)

    return response, model


def generate_promo_video_response(session_state):
    """Return the response from the promo video model.

    Args:
        session_state (dict): Streamlit session state
    Returns:
        response (dict): Response from LLM
        model (obj): Model object
    """

    # Load the config
    with open("./data/configs/vertexai_styleguide.json", "r") as fn:
        config = json.load(fn)

    # Initialize the model
    model = vertexai_promovideo.VertexAIPromoVideo(config)

    # Pass customer insights into model
    model.customer_insights = {
        "gender": session_state["gender"],
        "age": session_state["age"],
        "income": session_state["income"],
        "style": session_state["style"],
        "outerwear1": session_state["outerwear1"],
        "outerwear2": session_state["outerwear2"],
        "outerwear3": session_state["outerwear3"],
        "top1": session_state["top1"],
        "top2": session_state["top2"],
        "top3": session_state["top3"],
        "bottom1": session_state["bottom1"],
        "bottom2": session_state["bottom2"],
        "bottom3": session_state["bottom3"],
    }

    # Setup model
    model.use_default_template()
    model.setup_model()

    # Get response
    response = model.add_user_input(model.customer_insights)
    print(response)

    return response, model
