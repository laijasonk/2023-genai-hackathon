"""VertexAI API to iteratively create fashion recommendations.

Generative AI based on VertexAI's PaLM to return fashion info as it
chats with the user. 
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


class VertexAIFashion(ChatBot):
    """PaLM agent for generating iterative fashion recommendations.

    Attributes:
        history (list): Memory of the full conversation in string format
        recommendation (dict): Last recorded recommendations
    """

    history = []
    recommendation = {
        "clothing": "unknown",
        "color": "unknown",
        "pattern": "unknown",
        "sleeve": "unknown",
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
            self.identity = (
                "You are a fashion designer working for a company called Capgemini."
            )
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
            self.recommendation["clothing"] = output["clothing"]
            self.recommendation["color"] = output["color"]
            self.recommendation["pattern"] = output["pattern"]
            self.recommendation["sleeve"] = output["sleeve"]
            output["exception"] = False  # inform when the LLM misbehaved
        except:
            output = {"response": text.strip()}
            output["clothing"] = self.recommendation["clothing"]
            output["color"] = self.recommendation["color"]
            output["pattern"] = self.recommendation["pattern"]
            output["sleeve"] = self.recommendation["sleeve"]
            output["exception"] = True  # inform when the LLM misbehaved

        # Handle manually due to LangChain quirks
        self.history.append([user_input, output["response"]])
        self.memory = self._memory()
        for entry in self.history:
            user_input = (
                f"You recommended {output['clothing']} clothing in {output['color']} color, {output['pattern']} pattern, and {output['sleeve']} sleeves. "
                + entry[0]
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

        template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\n{{history}}\n{{format_instructions}}\n{{user_input}}\n"
        return template

    def _prompt(self):
        """Private method to define LangChain prompt template.

        Args:
            None
        Returns:
            prompt (obj): LangChain prompt template object
        """

        response_schemas = [
            ResponseSchema(
                name="response",
                description="Give the customer an ongoing clothing or fashion recommendation based on the conversation",
            ),
            ResponseSchema(
                name="clothing",
                description="Select one clothing recommendation from the list: dress, jacket, tshirt, casual-shirt, formal-shirt, skirt, blouse, unknown",
            ),
            ResponseSchema(
                name="color",
                description="Select one color recommendation from the list: white, black, blue, red, green, yellow, beige, unknown",
            ),
            ResponseSchema(
                name="pattern",
                description="Select one pattern recommendation from the list: solid, stripes, dotted, floral, unknown",
            ),
            ResponseSchema(
                name="sleeve",
                description="Select one sleeve recommendation from the list: short-sleeve, long-sleeve, no-sleeve, unknown",
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
