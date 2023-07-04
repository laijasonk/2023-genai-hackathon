"""VertexAI API to create a promo video.

Generative AI based on VertexAI's PaLM to create a promo video based on
the input information.
"""

import os
import sys
import logging
import json
import re

from langchain.embeddings import VertexAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

from google.cloud import texttospeech
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import *
from PIL import Image
import numpy

# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("../../../"))

from src.models import ChatBot


class VertexAIPromoVideo(ChatBot):
    """PaLM agent for generating promo videos."""

    customer_insights = {
        "gender": "",
        "age": "",
        "income": "",
        "style": "",
        "outerwear1": "",
        "outerwear2": "",
        "outerwear3": "",
        "top1": "",
        "top2": "",
        "top3": "",
        "bottom1": "",
        "bottom2": "",
        "bottom3": "",
    }

    def use_default_template(self):
        """Define the default prompt scenario.

        Args:
            None
        Returns:
            None
        """

        # Prepare prompt
        if not self.config.get("prompt", {}).get("identity", False):
            self.identity = "You are a fashion expert working a clothing company."
        else:
            self.identity = self.config["prompt"]["identity"]
        if not self.config.get("prompt", {}).get("intent", False):
            self.intent = "You write marketing material for your clothing company personalized to the customer."
        else:
            self.intent = self.config["prompt"]["intent"]
        if not self.config.get("prompt", {}).get("behavior", False):
            self.behavior = "You are well-spoken and clear."
        else:
            self.behavior = self.config["prompt"]["behavior"]

        # Set defaults
        self.template = self._template()

        return None

    def add_user_input(self, input_dict):
        """Append a text input to the chain.

        Args:
            user_input_dict (dict): Text input to append to chain
        Returns:
            output (dict): Formatted JSON response
        """

        try:
            user_input = f"""
The customer is {input_dict["age"]} years old and {input_dict["gender"]}.
The customer is in the {input_dict["income"]} income bracket group.
The customer prefers the {input_dict["style"]} fashion style.
The customer is considering buying {input_dict["outerwear1"]}, {input_dict["outerwear2"]}, or {input_dict["outerwear3"]} as outerwear.
The customer is considering buying {input_dict["top1"]}, {input_dict["top2"]}, or {input_dict["top3"]} as a top.
The customer is considering buying {input_dict["bottom1"]}, {input_dict["bottom2"]}, or {input_dict["bottom3"]} as bottoms.
"""
            response = self.chain(user_input)

            text = str(response["text"])
            text = re.sub("^```json", "", text)
            text = re.sub("```$", "", text)

            if text:
                output = json.loads(text)
            else:
                output = {}
        except:
            output = {}

        return output

    def _llm(self):
        """Private method to define LangChain LLM.

        Args:
            None
        Returns:
            llm (obj): LangChain LLM object
        """

        api_key = self.config.get("vertexai", {}).get("api_key", "")
        temperature = self.config.get("vertexai", {}).get("temperature", 0.9)
        model_name = self.config.get("vertexai", {}).get("model_name", "chat-bison")
        max_tokens = self.config.get("vertexai", {}).get("max_tokens", "2048")

        llm = ChatVertexAI(
            temperature=temperature,
            model_name=model_name,
            max_output_tokens=max_tokens,
        )

        return llm

    def _embeddings(self):
        """Private method to define LangChain embeddings.

        Args:
            None
        Returns:
            model (obj): Pre-trained model object
        """

        api_key = self.config.get("vertexai", {}).get("api_key", "")
        embeddings = VertexAIEmbeddings()
        return embeddings

    def _chain(self):
        """Private method to define LangChain chain.

        Args:
            None
        Returns:
            chain (obj): LangChain chain object
        """

        chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False,
        )
        return chain

    def _memory(self):
        """Private method to define LangChain memory.

        Args:
            None
        Returns:
            memory (obj): LangChain memory object
        """

        memory = None
        return memory

    def _template(self):
        """Private method to define text template.

        Args:
            None
        Returns:
            template (str): Text template for prompt
        """

        template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\n{{format_instructions}}{{question}}\n"
        return template

    def _prompt(self):
        """Private method to define LangChain prompt template.

        Args:
            None
        Returns:
            prompt (obj): LangChain prompt template object
        """

        info = self.customer_insights

        pitch = f"Write a short three sentence pitch from a clothing company that would appeal to a {info['age']} year old {info['gender']} customers with {info['income']} income and a preference for {info['style']} fashion."

        response_schemas = [
            ResponseSchema(
                name="pitch",
                description=pitch,
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt

    def generate_video(self, response, mp4_filename):
        """Generate a promo video based on response.

        Args:
            response (dict): Formatted JSON response
        Returns:
            None
        """

        script = response["pitch"]

        # Audio
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=script)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1
        )
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        with open("output.mp3", "wb") as out:
            out.write(response.audio_content)

        # loading audio file
        audioclip = AudioFileClip("./output.mp3")

        # Adjust length depending on audio
        fps = 30
        fx_speed = 2 / int(audioclip.duration)
        clip_length = int(int(audioclip.duration) / 3)
        crossfade = int(int(audioclip.duration) / 5)
        duration = clip_length + crossfade

        # Video
        txt1 = TextClip(
            f"{self.customer_insights['outerwear1']}\n{self.customer_insights['top1']}\n{self.customer_insights['bottom1']}",
            fontsize=24,
            color="black",
        )
        image_width, image_height = txt1.size
        txt1_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt1 = (
            txt1.set_pos("center")
            .margin(bottom=100, right=150, opacity=0)
            .set_duration(duration)
        )
        txt1_box = (
            txt1_box.set_opacity(0.5)
            .set_pos("center")
            .margin(bottom=100, right=150, opacity=0)
            .set_duration(duration)
        )

        img1 = Image.open("data/templates/clothing1.png")
        img1_array = numpy.array(img1)[:, :, :3]

        def clip1_fun(t):
            return numpy.roll(img1_array, int(t * fx_speed * fps), axis=1)

        clip1 = CompositeVideoClip(
            [VideoClip(clip1_fun, duration=duration), txt1_box, txt1]
        )
        clip1.fps = fps

        txt2 = TextClip(
            f"{self.customer_insights['outerwear2']}\n{self.customer_insights['top2']}\n{self.customer_insights['bottom2']}",
            fontsize=24,
            color="black",
        )
        image_width, image_height = txt2.size
        txt2_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt2 = txt2.set_pos("center").set_duration(duration)
        txt2_box = txt2_box.set_opacity(0.5).set_pos("center").set_duration(duration)

        img2 = Image.open("data/templates/clothing2.png")
        img2_array = numpy.array(img2)[:, :, :3]

        def clip2_fun(t):
            return numpy.roll(img2_array, -int(t * fx_speed * fps), axis=1)

        clip2 = CompositeVideoClip(
            [VideoClip(clip2_fun, duration=duration), txt2_box, txt2]
        )
        clip2.fps = fps

        txt3 = TextClip(
            f"{self.customer_insights['outerwear3']}\n{self.customer_insights['top3']}\n{self.customer_insights['bottom3']}",
            fontsize=24,
            color="black",
        )
        image_width, image_height = txt3.size
        txt3_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt3 = (
            txt3.set_pos("center")
            .margin(top=100, left=150, opacity=0)
            .set_duration(duration)
        )
        txt3_box = (
            txt3_box.set_opacity(0.5)
            .set_pos("center")
            .margin(top=100, left=150, opacity=0)
            .set_duration(duration)
        )

        img3 = Image.open("data/templates/clothing3.png")
        img3_array = numpy.array(img3)[:, :, :3]

        def clip3_fun(t):
            return numpy.roll(img3_array, int(t * fx_speed * fps), axis=1)

        clip3 = CompositeVideoClip(
            [VideoClip(clip3_fun, duration=duration), txt3_box, txt3]
        )
        clip3.fps = fps

        clip = concatenate(
            [
                clip1.crossfadein(crossfade),
                clip2.crossfadein(crossfade),
                clip3.crossfadein(crossfade),
            ],
            padding=-crossfade,
            method="compose",
        )

        # adding audio to the video clip
        videoclip = clip.set_audio(audioclip)

        # write to disk
        videoclip.write_videofile(mp4_filename)

        return None
