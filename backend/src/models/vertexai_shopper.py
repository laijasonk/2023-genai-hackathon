"""VertexAI API to interactively give clothing recommendations.

Generative AI based on VertexAI's PaLM to return clothes shopping
recommendations as it chats with the user. 
"""

import os
import sys
import logging
import json
import re

from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema


# Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
sys.path.append(os.path.abspath("../../../"))

from src.models import ChatBot


class VertexAIShopper(ChatBot):
    """PaLM agent for generating iterative clothing recommendations.

    Attributes:
        history (list): Memory of the full conversation in string format
        recommendation (dict): Last recorded recommendations
    """

    history = []
    recommendation = {
        "outerwear": [],
        "outerwear_color": [],
        "top": [],
        "top_color": [],
        "bottom": [],
        "bottom_color": [],
        "customer_gender": None,
        "customer_age": None,
        "customer_income": None,
        "customer_style": None,
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
            self.identity = "You are a personal clothing shopper working for a company called Capgemini."
        else:
            self.identity = self.config["prompt"]["identity"]
        if not self.config.get("prompt", {}).get("intent", False):
            self.intent = "You make recommendations on clothing based on your chat with the customer."
        else:
            self.intent = self.config["prompt"]["intent"]
        if not self.config.get("prompt", {}).get("behavior", False):
            self.behavior = "You are verbose, logical, and well-spoken."
        else:
            self.behavior = self.config["prompt"]["behavior"]

        # Set defaults
        self.template = self._template()

        return None

    def add_user_input(self, user_input):
        """Append a text input to the chain.

        Args:
            user_input (str): Text input to append to chain
        Returns:
            output (dict): Formatted JSON response
        """

        user_input = str(user_input).strip()
        memory = self.memory.buffer  # self.memory.load_memory_variables({})["history"]
        response = self.chain(
            {self.memory.input_key: user_input, self.memory.memory_key: memory}
        )

        text = str(response["text"])
        text = re.sub("^```json", "", text)
        text = re.sub("```$", "", text)

        # Non-elegant way of handling when the LLM misbehaves...
        try:
            output = json.loads(text.strip())
            self.recommendation["outerwear"] = output["outerwear"]
            self.recommendation["outerwear_color"] = output["outerwear_color"]
            self.recommendation["top"] = output["top"]
            self.recommendation["top_color"] = output["top_color"]
            self.recommendation["bottom"] = output["bottom"]
            self.recommendation["bottom_color"] = output["bottom_color"]
            self.recommendation["customer_gender"] = output["customer_gender"]
            self.recommendation["customer_age"] = output["customer_age"]
            self.recommendation["customer_income"] = output["customer_income"]
            self.recommendation["customer_style"] = output["customer_style"]
            output["exception"] = False  # inform when the LLM misbehaved
        except:
            output = {"response": text.strip()}
            output["outerwear"] = self.recommendation["outerwear"]
            output["outerwear_color"] = self.recommendation["outerwear_color"]
            output["top"] = self.recommendation["top"]
            output["top_color"] = self.recommendation["top_color"]
            output["bottom"] = self.recommendation["bottom"]
            output["bottom_color"] = self.recommendation["bottom_color"]
            output["customer_gender"] = self.recommendation["customer_gender"]
            output["customer_age"] = self.recommendation["customer_age"]
            output["customer_income"] = self.recommendation["customer_income"]
            output["customer_style"] = self.recommendation["customer_style"]
            output["exception"] = True  # inform when the LLM misbehaved

        # Handle manually due to LangChain quirks
        self.history.append([user_input, output["response"]])
        self.memory = self._memory()
        for entry in self.history:
            outerwear = ""
            outerwear = (
                self.config["prompt"]
                .get("outerwear", {})
                .get(str(output["outerwear"]), "none")
            )
            outerwear_color = (
                self.config["prompt"]
                .get("outerwear_color", {})
                .get(str(output["outerwear_color"]), "white")
            )
            top = (
                self.config["prompt"].get("top", {}).get(str(output["top"]), "t-shirt")
            )
            top_color = (
                self.config["prompt"]
                .get("top_color", {})
                .get(str(output["top_color"]), "white")
            )
            bottom = (
                self.config["prompt"]
                .get("bottom", {})
                .get(str(output["bottom"]), "shorts")
            )
            bottom_color = (
                self.config["prompt"]
                .get("bottom_color", {})
                .get(str(output["bottom_color"]), "white")
            )

            prompt_outerwear = ""
            prompt_top = ""
            prompt_bottom = ""

            if not outerwear == "none":
                prompt_outerwear = f"{outerwear_color} {outerwear} over "

            prompt_top = f"{top_color} {top}"
            if not str(top) == "dress":
                bottom = f"and {bottom_color} {bottom}"

            user_input = (
                # f"I was recommended the following clothing: {prompt_outerwear}{prompt_top}{prompt_bottom}. " +
                entry[0]
            )
            response = entry[1]
            self.memory.save_context(
                {self.memory.input_key: user_input}, {self.memory.output_key: response}
            )

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
            # memory=self.memory,  # handle manually due to LangChain quirks
        )
        return chain

    def _memory(self):
        """Private method to define LangChain memory.

        Args:
            None
        Returns:
            memory (obj): LangChain memory object
        """

        memory = ConversationBufferMemory(
            input_key="user_input",
            output_key="text",
            memory_key="history",
            return_messages=True,
        )
        self.conversation = []
        return memory

    def _template(self):
        """Private method to define text template.

        Args:
            None
        Returns:
            template (str): Text template for prompt
        """

        template = f"{self.identity}\n{self.intent}\n{self.behavior}\n---------------\n{{history}}\n---------------\n{{format_instructions}}\n{{user_input}}\n"
        return template

    def _prompt(self):
        """Private method to define LangChain prompt template.

        Args:
            None
        Returns:
            prompt (obj): LangChain prompt template object
        """

        response = self.config.get("prompt", {}).get("response", "")
        outerwear = self.config.get("prompt", {}).get("outerwear", {})
        outerwear_color = self.config.get("prompt", {}).get("outerwear_color", {})
        top = self.config.get("prompt", {}).get("top", {})
        top_color = self.config.get("prompt", {}).get("top_color", {})
        bottom = self.config.get("prompt", {}).get("bottom", {})
        bottom_color = self.config.get("prompt", {}).get("bottom_color", {})
        customer_gender = self.config.get("prompt", {}).get("customer_gender", "")
        customer_age = self.config.get("prompt", {}).get("customer_age", "")
        customer_income = self.config.get("prompt", {}).get("customer_income", "")
        customer_style = self.config.get("prompt", {}).get("customer_style", "")

        response_schemas = [
            ResponseSchema(
                name="response",
                description=response,
                type="string",
            ),
            ResponseSchema(
                name="outerwear",
                description=self._build_prompt(outerwear),
                type="list",
            ),
            ResponseSchema(
                name="outerwear_color",
                description=self._build_prompt(outerwear_color),
                type="list",
            ),
            ResponseSchema(
                name="top",
                description=self._build_prompt(top),
                type="list",
            ),
            ResponseSchema(
                name="top_color",
                description=self._build_prompt(top_color),
                type="list",
            ),
            ResponseSchema(
                name="bottom",
                description=self._build_prompt(bottom),
                type="list",
            ),
            ResponseSchema(
                name="bottom_color",
                description=self._build_prompt(bottom_color),
                type="list",
            ),
            ResponseSchema(
                name="customer_gender",
                description=self._build_prompt(customer_gender),
                type="string",
            ),
            ResponseSchema(
                name="customer_age",
                description=self._build_prompt(customer_age),
                type="string",
            ),
            ResponseSchema(
                name="customer_income",
                description=self._build_prompt(customer_income),
                type="string",
            ),
            ResponseSchema(
                name="customer_style",
                description=self._build_prompt(customer_style),
                type="string",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["history", "user_input"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt

    def _build_prompt(self, prompt_parts):
        """Prive method to build prompts from parts from config.

        Args:
            prompt_parts (dict): Prompt sections
        Returns:
            None
        """

        prompt_options = "Consider the following as examples: {prompt_parts[1]}"
        option = 2
        while str(option) in prompt_parts:
            prompt_options += f", {prompt_parts[str(option)]}"
            option += 1
        prompt = (
            f"{prompt_parts['leading']} {prompt_options}. {prompt_parts['trailing']}"
        )

        # prompt_options = f"Using 0 for {prompt_parts['0']}"
        # option = 1
        # while str(option) in prompt_parts:
        #     prompt_options += f", {option} for {prompt_parts[str(option)]}"
        #     option += 1
        # prompt = f"{prompt_parts['leading']} {prompt_options}, {prompt_parts['trailing']}"

        return prompt

    def add_streamlit_history(self, past=[], generated=[]):
        """Append streamlit history to LangChain memory object.

        Args:
            past (list): streamlit_chat's past session list
            generated (list): streamlit_chat's generated session list
        Returns:
            None
        """

        self.history = []
        for idx in range(len(generated)):
            self.history.append([past[idx], generated[idx]])
            self.memory.save_context(
                {self.memory.input_key: past[idx]},
                {self.memory.output_key: generated[idx]},
            )
        return None
