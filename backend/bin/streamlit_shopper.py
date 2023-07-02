import os
import sys
import json

import streamlit as st
from streamlit_chat import message

# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("./"))
from src.models import vertexai_shopper

# Load the config
with open("./data/configs/vertexai_shopper.json", "r") as fn:
    config = json.load(fn)

# Config streamlit
st.set_page_config(
    page_title="Personal Shopper",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Initialize the chatbot
chatbot = vertexai_shopper.VertexAIShopper(config)

# Functions
def generate_response(user_input):
    chatbot.add_streamlit_history(st.session_state["past"], st.session_state["generated"])
    response = chatbot.add_user_input(user_input)
    print(response)
    return response

def get_text():
    # input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    input_text = st.text_input("You: ", key="input", on_change=on_submit_text)
    return st.session_state["prompt"]

def on_submit_text():
    st.session_state["prompt"] = st.session_state["input"]
    st.session_state["input"] = ""
    return None

def set_recommendations(response):
    _validate_recommendation(response, "outerwear")
    _validate_recommendation(response, "top")
    _validate_recommendation(response, "bottom")
    if "dress" in st.session_state["top1"]: st.session_state["bottom1"] = st.session_state["top1"]
    if "dress" in st.session_state["top2"]: st.session_state["bottom2"] = st.session_state["top2"]
    if "dress" in st.session_state["top3"]: st.session_state["bottom3"] = st.session_state["top3"]
    if "dress" in st.session_state["bottom1"]: st.session_state["top1"] = st.session_state["bottom1"]
    if "dress" in st.session_state["bottom2"]: st.session_state["top2"] = st.session_state["bottom2"]
    if "dress" in st.session_state["bottom3"]: st.session_state["top3"] = st.session_state["bottom3"]
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

        st.session_state[clothing + "1"] = f"{response[clothing + '_color'][0]} {response[clothing][0]}"
        st.session_state[clothing + "2"] = f"{response[clothing + '_color'][0]} {response[clothing][1]}"
        st.session_state[clothing + "3"] = f"{response[clothing + '_color'][1]} {response[clothing][0]}"
    except:
        st.session_state[clothing + "1"] = ""
        st.session_state[clothing + "2"] = ""
        st.session_state[clothing + "3"] = ""
    
    return None

def set_demographics(response):
    _validate_demographics(response, "gender")
    _validate_demographics(response, "age")
    _validate_demographics(response, "income")
    _validate_demographics(response, "style")
    return None

def _validate_demographics(response, demographics):
    try:
        st.session_state[demographics] = response["customer_" + demographics]
    except:
        st.session_state[demographics] = ""
    
    return None

# Creating the chatbot interface
st.title("Personal Clothing Shopper")

# Storing the chat
if "past" not in st.session_state: st.session_state["past"] = []
if "generated" not in st.session_state: st.session_state["generated"] = []
if "prompt" not in st.session_state: st.session_state["prompt"] = ""

if "outerwear1" not in st.session_state: st.session_state["outerwear1"] = ""
if "outerwear2" not in st.session_state: st.session_state["outerwear2"] = ""
if "outerwear3" not in st.session_state: st.session_state["outerwear3"] = ""
if "top1" not in st.session_state: st.session_state["top1"] = ""
if "top2" not in st.session_state: st.session_state["top2"] = ""
if "top3" not in st.session_state: st.session_state["top3"] = ""
if "bottom1" not in st.session_state: st.session_state["bottom1"] = ""
if "bottom2" not in st.session_state: st.session_state["bottom2"] = ""
if "bottom3" not in st.session_state: st.session_state["bottom3"] = ""

if "gender" not in st.session_state: st.session_state["gender"] = ""
if "age" not in st.session_state: st.session_state["age"] = ""
if "income" not in st.session_state: st.session_state["income"] = ""
if "style" not in st.session_state: st.session_state["style"] = ""

clothing1, clothing2, clothing3 = st.columns(3)
recommendation1 = clothing1.text(f"Outer:\nTop:\nBottom:")
recommendation2 = clothing2.text(f"Outer:\nTop:\nBottom:")
recommendation3 = clothing3.text(f"Outer:\nTop:\nBottom:")

st.sidebar.title("Customer Insights")
demographics1 = st.sidebar.text(f"Guessed Gender:")
demographics2 = st.sidebar.text(f"Guessed Age:")
demographics3 = st.sidebar.text(f"Guessed Income:")
demographics4 = st.sidebar.text(f"Guessed Style:")

user_input = get_text()

if user_input:
    response = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response["response"])
    st.session_state["text"] = ""
    
    set_recommendations(response)
    recommendation1.text(f"Outer: {st.session_state['outerwear1']}\nTop: {st.session_state['top1']}\nBottom: {st.session_state['bottom1']}")
    recommendation2.text(f"Outer: {st.session_state['outerwear2']}\nTop: {st.session_state['top2']}\nBottom: {st.session_state['bottom2']}")
    recommendation3.text(f"Outer: {st.session_state['outerwear3']}\nTop: {st.session_state['top3']}\nBottom: {st.session_state['bottom3']}")

    set_demographics(response)
    demographics1.text(f"Guessed Gender: {st.session_state['gender']}")
    demographics2.text(f"Guessed Age: {st.session_state['age']}")
    demographics3.text(f"Guessed Income: {st.session_state['income']}")
    demographics4.text(f"Guessed Style: {st.session_state['style']}")
else:
    st.session_state.past.append("Hi")
    st.session_state.generated.append("I am your personal clothing shopper! How can I help you?")

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], avatar_style="avataaars", seed=1234, key=str(i))
        if not i == 0:
            message(st.session_state["past"][i], avatar_style="icons", seed=1234, is_user=True, key=str(i) + "_user")

