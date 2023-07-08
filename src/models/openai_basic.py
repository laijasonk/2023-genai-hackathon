# """OpenAI API to generate a basic conversational chatbot.
#
# Chatbot based on OpenAI's ChatGPT without providing any additional
# context to the agent.
# """
#
# import os
# import sys
# import logging
#
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from langchain import PromptTemplate
#
# # Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
# sys.path.append(os.path.abspath("../../../"))
#
# from src.models import ChatBot
#
#
# class OpenAIBasic(ChatBot):
#     """Basic ChatGPT agent."""
#
#     def use_default_template(self):
#         """Define the default prompt scenario.
#
#         Args:
#             None
#         Returns:
#             None
#         """
#
#         # Prepare prompt
#         if not self.config.get("prompt", {}).get("identity", False):
#             self.identity = "You are an AI assistent for a company called Capgemini"
#         else:
#             self.identity = self.config["prompt"]["identity"]
#         if not self.config.get("prompt", {}).get("intent", False):
#             self.intent = (
#                 "You are answering questions in a conversation with an employee."
#             )
#         else:
#             self.intent = self.config["prompt"]["intent"]
#         if not self.config.get("prompt", {}).get("behavior", False):
#             self.behavior = "You are conversational, helpful, and based on the knowledge and facts provided."
#         else:
#             self.behavior = self.config["prompt"]["behavior"]
#
#         # Set defaults
#         self.template = self._template()
#
#         return None
#
#     def _llm(self):
#         """Private method to define LangChain LLM.
#
#         Args:
#             None
#         Returns:
#             llm (obj): LangChain LLM object
#         """
#
#         api_key = self.config.get("openai", {}).get("api_key", "")
#         temperature = self.config.get("openai", {}).get("temperature", 0.9)
#         model_name = self.config.get("openai", {}).get("model_name", "gpt-3.5-turbo")
#         max_tokens = self.config.get("openai", {}).get("max_tokens", "256")
#
#         llm = ChatOpenAI(
#             temperature=temperature,
#             model_name=model_name,
#             max_tokens=max_tokens,
#             openai_api_key=api_key,
#         )
#
#         return llm
#
#     def _embeddings(self):
#         """Private method to define LangChain embeddings.
#
#         Args:
#             None
#         Returns:
#             model (obj): Pre-trained model object
#         """
#
#         api_key = self.config.get("openai", {}).get("api_key", "")
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#         return embeddings
#
#     def _chain(self):
#         """Private method to define LangChain chain.
#
#         Args:
#             None
#         Returns:
#             chain (obj): LangChain chain object
#         """
#
#         chain = LLMChain(
#             llm=self.llm,
#             prompt=self.prompt,
#             memory=self.memory,
#             verbose=False,
#         )
#         return chain
#
#     def _memory(self):
#         """Private method to define LangChain memory.
#
#         Args:
#             None
#         Returns:
#             memory (obj): LangChain memory object
#         """
#
#         memory = ConversationBufferMemory(
#             input_key="question",
#             output_key="text",
#             memory_key="chat_history",
#             return_messages=True,
#         )
#         return memory
#
#     def _template(self):
#         """Private method to define text template.
#
#         Args:
#             None
#         Returns:
#             template (str): Text template for prompt
#         """
#
#         template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\n{{chat_history}}\n{{question}}\nAI: "
#         return template
#
#     def _prompt(self):
#         """Private method to define LangChain prompt template.
#
#         Args:
#             None
#         Returns:
#             prompt (obj): LangChain prompt template object
#         """
#
#         prompt = PromptTemplate(
#             input_variables=["chat_history", "question"],
#             template=self.template,
#         )
#
#         return prompt
