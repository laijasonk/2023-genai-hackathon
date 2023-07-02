"""Modules to initialize AI chatbots.

Base class for AI chatbots to inherit from. This class essentially acts as
a template so that all chatbots can be plug-and-play (modular).
"""

import os
import sys
import logging


class ChatBot:
    """Base class to structure AI chatbot code.

    Attributes:
        config (dict): Model-specific parameters
        chain (obj): LangChain chain object
        llm (obj): LangChain LLM object
        embeddings (obj): LangChain embeddings object
        memory (obj): LangChain memory object
        prompt (obj): LangChain prompt template object
        identity (str): Prompt engineering chatbot's identity
        intent (str): Prompt engineering chatbot's intent
        behavior (str): Prompt engineering chatbot's behavior
        template (str): Prompt template
    """

    def __init__(self, config={}):
        """Default constructor.

        Args:
            config (dict): Model-specific parameters
        Returns:
            None
        """

        self.config = config
        try:
            self.use_default_template()
            self.setup_model()
        except:
            pass

        return None

    def setup_model(
        self,
        embeddings=None,
        llm=None,
        memory=None,
        template="",
        prompt=None,
        chain=None,
    ):
        """Load or fetch objects for model.

        Args:
            embeddings (obj): LangChain embeddings object
            llm (obj): LangChain LLM object
            chain (obj): LangChain chain object
        Returns:
            None
        """

        if not embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = self._embeddings()

        if not llm is None:
            self.llm = llm
        else:
            self.llm = self._llm()

        if not memory is None:
            self.memory = memory
        else:
            self.memory = self._memory()

        if not template == "":
            self.template = template
        else:
            self.template = self._template()

        if not prompt is None:
            self.prompt = prompt
        else:
            self.prompt = self._prompt()

        if not chain is None:
            self.chain = chain
        else:
            self.chain = self._chain()

        return None

    def set_prompt_template(self, identity=None, intent=None, behavior=None):
        """Set the prompt template via common parts.

        Args:
            identity (str): Prompt engineering chatbot's identity
            intent (str): Prompt engineering chatbot's intent
            behavior (str): Prompt engineering chatbot's behavior
        Returns:
            None
        """

        if identity:
            self.identity = identity.strip()
        if intent:
            self.intent = intent.strip()
        if behavior:
            self.behavior = behavior.strip()

        self.template = self._template()

        return None

    def use_default_template(self):
        """Define the default prompt scenario.

        Args:
            None
        Returns:
            None
        """

        if not self.config.get("prompt", {}).get("identity", False):
            self.identity = "You are an AI assistent for a company called Capgemini Applied Innovation Exchange San Francisco. "
        else:
            self.identity = self.config["prompt"]["identity"]
        if not self.config.get("prompt", {}).get("intent", False):
            self.intent = (
                "You are answering questions in a conversation with an employee. "
            )
        else:
            self.intent = self.config["prompt"]["intent"]
        if not self.config.get("prompt", {}).get("behavior", False):
            self.behavior = "You are conversational, helpful, and friendly. "
        else:
            self.behavior = self.config["prompt"]["behavior"]

        # Set defaults
        self.set_prompt_template(identity, intent, behavior)

        return None

    def add_user_input(self, user_input):
        """Append a text input to the chain.

        Args:
            user_input (str): Text input to append to chain
        Returns:
            response (str): Chatbot response to user input
        """

        user_input = str(user_input).strip()
        response = self.chain({self.memory.input_key: user_input})

        return response.get(self.memory.output_key)

    def add_history(self, history):
        """Append list of history to LangChain memory object.

        Args:
            history (list): List of past prompts and responses
        Returns:
            None
        """

        for entry in history:
            user_input = list(entry.keys())[0]
            response = list(entry.values())[0]
            self.memory.save_context(
                {self.memory.input_key: user_input}, {self.memory.output_key: response}
            )

        return None

    def get_history(self):
        """Generate text based on current prompt.

        Args:
            None
        Returns:
            history (list): List of past input/outputs in dict format
        """

        history = []
        count = 0
        for entry in self.memory.load_memory_variables({})["chat_history"]:
            count += 1
            if count % 2 == 1:
                user_input = entry.content
            else:
                history.append({user_input: entry.content})

        return history

    def _llm(self):
        """Private method to define LangChain LLM.

        Args:
            None
        Returns:
            llm (obj): LangChain LLM object
        """

        return None

    def _embeddings(self):
        """Private method to define LangChain embeddings.

        Args:
            None
        Returns:
            embeddings (obj): LangChain embeddings object
        """

        return None

    def _chain(self):
        """Private method to define LangChain chain.

        Args:
            None
        Returns:
            chain (obj): LangChain chain object
        """

        return None

    def _memory(self):
        """Private method to define LangChain memory.

        Args:
            None
        Returns:
            memory (obj): LangChain memory object
        """

        return None

    def _template(self):
        """Private method to define text template.

        Args:
            None
        Returns:
            template (str): Text template for prompt
        """

        template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\n{{context}}\n----------------\n{{history}}\n{{question}}\nAI: "
        return template

    def _prompt(self):
        """Private method to define LangChain prompt.

        Args:
            None
        Returns:
            prompt (obj): LangChain prompt template object
        """

        return None


__all__ = [
    "vertexai_basic",
    "vertexai_shopper",
    "vertexai_styleguide",
]
