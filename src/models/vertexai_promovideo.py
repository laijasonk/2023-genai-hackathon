"""VertexAI API to create a promo video.

Generative AI based on VertexAI's PaLM to create a promo video based on
the input information.
"""

import os
import sys
import logging
import json
import re
import math
import numpy

from langchain.embeddings import VertexAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

from google.cloud import aiplatform
from google.cloud import texttospeech

from moviepy.editor import *
import moviepy.video.io.ImageSequenceClip

from PIL import Image
from io import BytesIO
import base64


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

        pitch = f"Write a short two sentence pitch from a clothing company that would appeal to a {info['age']} year old {info['gender']} customers with {info['income']} income and a preference for {info['style']} fashion."

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

        img1_array, img2_array, img3_array = self.generate_images()

        script = response["pitch"]
        speaking_rate = 1.5

        # Audio
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=script)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
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
        clip_length = math.ceil(math.ceil(audioclip.duration) / 3.0)
        crossfade = int(int(audioclip.duration) / 5)
        duration = clip_length + crossfade

        # Video
        txt1 = TextClip(
            # f"{self.customer_insights['outerwear1']}\n{self.customer_insights['top1']}\n{self.customer_insights['bottom1']}",
            f"{self.customer_insights['outerwear1']}, {self.customer_insights['top1']},\n{self.customer_insights['bottom1']}",
            fontsize=24,
            color="black",
        )
        image_width, image_height = txt1.size
        txt1_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt1 = (
            txt1
            # .margin(bottom=100, right=150, opacity=0)
            .set_pos("center").margin(top=350, opacity=0)
            # .set_position((0.0, 0.40), relative=True)
            .set_duration(duration)
        )
        txt1_box = (
            txt1_box.set_opacity(0.75)
            .set_pos("center")
            .margin(top=350, opacity=0)
            # .margin(bottom=100, right=150, opacity=0)
            # .set_position((0.0, 0.40), relative=True)
            .set_duration(duration)
        )

        # img1 = Image.open("data/templates/clothing1.png")
        # img1_array = numpy.array(img1)[:, :, :3]

        def clip1_fun(t):
            return numpy.roll(img1_array, int(t * fx_speed * fps), axis=1)

        clip1 = CompositeVideoClip(
            [VideoClip(clip1_fun, duration=duration), txt1_box, txt1]
        )
        clip1.fps = fps

        txt2 = TextClip(
            # f"{self.customer_insights['outerwear2']}\n{self.customer_insights['top2']}\n{self.customer_insights['bottom2']}",
            f"{self.customer_insights['outerwear2']}, {self.customer_insights['top2']},\n{self.customer_insights['bottom2']}",
            fontsize=20,
            color="black",
        )
        image_width, image_height = txt2.size
        txt2_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt2 = (
            txt2
            # .set_pos("center", "bottom")
            .set_pos("center")
            .margin(top=350, opacity=0)
            .set_duration(duration)
        )
        txt2_box = (
            txt2_box.set_opacity(0.75)
            .set_pos("center")
            .margin(top=350, opacity=0)
            # .set_pos("center", "bottom")
            .set_duration(duration)
        )

        # img2 = Image.open("data/templates/clothing2.png")
        # img2_array = numpy.array(img2)[:, :, :3]

        def clip2_fun(t):
            return numpy.roll(img2_array, -int(t * fx_speed * fps), axis=1)

        clip2 = CompositeVideoClip(
            [VideoClip(clip2_fun, duration=duration), txt2_box, txt2]
        )
        clip2.fps = fps

        txt3 = TextClip(
            # f"{self.customer_insights['outerwear3']}\n{self.customer_insights['top3']}\n{self.customer_insights['bottom3']}",
            f"{self.customer_insights['outerwear3']}, {self.customer_insights['top3']},\n{self.customer_insights['bottom3']}",
            fontsize=20,
            color="black",
        )
        image_width, image_height = txt3.size
        txt3_box = ColorClip(
            size=(image_width + 20, image_height + 20), color=(200, 200, 230)
        )
        txt3 = (
            txt3
            # .set_pos("center", "bottom")
            # .margin(top=100, left=150, opacity=0)
            .set_pos("center")
            .margin(top=350, opacity=0)
            .set_duration(duration)
        )
        txt3_box = (
            txt3_box.set_opacity(0.75)
            .set_pos("center")
            .margin(top=350, opacity=0)
            # .set_pos("center", "bottom")
            # .margin(top=100, left=150, opacity=0)
            .set_duration(duration)
        )

        # img3 = Image.open("data/templates/clothing3.png")
        # img3_array = numpy.array(img3)[:, :, :3]

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

    def generate_images(self):
        """Generate a images for promo video.

        Args:
            None
        Returns:
            image1 (numpy): Array of generated image
            image2 (numpy): Array of generated image
            image3 (numpy): Array of generated image
        """

        # GCP details
        PROJECT_ID = "gen-hybrid-intelligence-team-1"
        REGION = "us-central1"  # @param {type:"string"}
        GCS_BUCKET = "model_storage_bucket_jeremy"
        ENDPOINT = 'projects/47448272174/locations/us-central1/endpoints/5040438378655383552'
        
        # Init system
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=GCS_BUCKET)
        ImageEndpoint = aiplatform.Endpoint(ENDPOINT)

        # Helper functions
        def image_to_base64(image, format="JPEG"):
            buffer = BytesIO()
            image.save(buffer, format=format)
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return image_str

        def base64_to_image(image_str):
            image = Image.open(BytesIO(base64.b64decode(image_str)))
            return image
            
        def text_guided_image_inpainting(prompt: str, init_image: Image.Image, mask_image: Image.Image, num_inference_steps=50) -> Image.Image:
            init_image = init_image.resize((512, 512))
            mask_image = mask_image.resize((512, 512))
            instances = [
                {
                    "prompt": prompt,
                    "image": image_to_base64(init_image),
                    "mask_image": image_to_base64(mask_image),
                    "num_inference_steps": num_inference_steps
                },
            ]
            response = ImageEndpoint.predict(instances=instances)
            images = [base64_to_image(image) for image in response.predictions]
            new_image = init_image.copy()
            new_image.paste(images[0], mask=mask_image.convert('L'))
            return new_image

        model_image = Image.open("./data/templates/model_photo.png").convert("RGB")
        rectangle_mask = Image.open("./data/templates/rectangle_mask.png").convert("L")

        images = []
        for idx in range(1, 4):
            if self.customer_insights[f'bottom{idx}'][-5:] == "dress":
                out = self.customer_insights[f"outerwear{idx}"]
                bot = self.customer_insights[f"bottom{idx}"]
                prompt = f"Outerwear: {out}. BREAK Underneath: {bot}."
            else:
                out = self.customer_insights[f"outerwear{idx}"]
                top = self.customer_insights[f"top{idx}"]
                bot = self.customer_insights[f"bottom{idx}"]
                prompt = f"Outerwear: {out}. BREAK Underneath: {top}. BREAK Bottom: {bot}."
            print(prompt)

            # Base run
            iteration1 = text_guided_image_inpainting(
                prompt=prompt,
                init_image=model_image,
                mask_image=rectangle_mask,
            )

            # Repeat to increase accuracy
            iteration2 = text_guided_image_inpainting(
                prompt=prompt,
                init_image=iteration1,
                mask_image=rectangle_mask,
            )

            # # Repeat to increase accuracy
            # iteration3 = text_guided_image_inpainting(
            #     prompt=prompt,
            #     init_image=iteration2,
            #     mask_image=rectangle_mask,
            # )

            images.append(iteration2)
            # images.append(iteration3)
        
        return images[0], images[1], images[2]

