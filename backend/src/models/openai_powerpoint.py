# """OpenAI API to generate a powerpoint proposal.
# 
# Generative AI based on OpenAI's ChatGPT to create a powerpoint proposal
# based on a single prompt.  """
# 
# import os
# import sys
# import logging
# import json
# import re
# 
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser
# from langchain.output_parsers import ResponseSchema
# 
# import collections
# import collections.abc
# from pptx import Presentation
# 
# 
# # Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
# sys.path.append(os.path.abspath("../../../"))
# 
# from src.models import ChatBot
# 
# 
# class OpenAIPowerpoint(ChatBot):
#     """ChatGPT agent for generating powerpoint proposals."""
# 
#     title = ""
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
#             self.identity = "You work for a company called Capgemini."
#         else:
#             self.identity = self.config["prompt"]["identity"]
#         if not self.config.get("prompt", {}).get("intent", False):
#             self.intent = "You write project proposals based on the input prompt."
#         else:
#             self.intent = self.config["prompt"]["intent"]
#         if not self.config.get("prompt", {}).get("behavior", False):
#             self.behavior = "You are verbose, logical, and well-spoken."
#         else:
#             self.behavior = self.config["prompt"]["behavior"]
# 
#         # Set defaults
#         self.template = self._template()
# 
#         return None
# 
#     def add_user_input(self, user_input):
#         """Append a text input to the chain.
# 
#         Args:
#             user_input (str): Text input to append to chain
#         Returns:
#             output (dict): Formatted JSON response
#         """
# 
#         user_input = str(user_input).strip()
#         response = self.chain(user_input)
# 
#         text = str(response["text"])
#         text = re.sub("^```json", "", text)
#         text = re.sub("```$", "", text)
# 
#         output = json.loads(text)
#         self.title = output["title"]
# 
#         return output
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
#         temperature = self.config.get("openai", {}).get("temperature", 0.0)
#         model_name = self.config.get("openai", {}).get("model_name", "gpt-3.5-turbo")
#         max_tokens = self.config.get("openai", {}).get("max_tokens", "2048")
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
#         memory = None
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
#         template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\n{{format_instructions}}{{question}}\n"
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
#         response_schemas = [
#             ResponseSchema(
#                 name="title",
#                 description="Write a short title with less than 10 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="executive",
#                 description="Write an abstract summary no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="objectives",
#                 description="Write the project objectives of a proposal no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="requirements",
#                 description="Write the project requirements of a proposal no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="phases",
#                 description="Write the project phases of a proposal no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="costs",
#                 description="Write the required effort and costs of a project proposal no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="deliverables",
#                 description="Write the expected deliverables of a project proposal no more than 200 words based on the user input.",
#             ),
#             ResponseSchema(
#                 name="acceptance",
#                 description="Write the acceptance criteria of a project proposal no more than 200 words based on the user input.",
#             ),
#         ]
#         output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 
#         format_instructions = output_parser.get_format_instructions()
#         prompt = PromptTemplate(
#             template=self.template,
#             input_variables=["question"],
#             partial_variables={"format_instructions": format_instructions},
#         )
# 
#         return prompt
# 
#     def generate_powerpoint(self, response, pptx_filename):
#         """Generate a powerpoint slide deck based on response.
# 
#         Args:
#             response (dict): Formatted JSON response
#         Returns:
#             None
#         """
# 
#         # Load a template
#         slides = Presentation("./data/templates/proposal.pptx")
# 
#         # Title
#         slide = slides.slides[0]
#         slide.shapes.title.text = response["title"]
#         slide.placeholders[1].text = "Sample slide deck created by Neo"
# 
#         # Executive Summary
#         slide = slides.slides[3]
#         slide.shapes[0].text = response["executive"]
# 
#         # Objectives and Requirements
#         slide = slides.slides[5]
#         slide.shapes[0].text = response["objectives"]
#         slide.shapes[3].text = response["requirements"]
# 
#         # Proposed Work Plan
#         slide = slides.slides[7]
#         slide.shapes[0].text = response["phases"]
#         slide.shapes[3].text = response["costs"]
# 
#         # Plan Operation
#         slide = slides.slides[9]
#         slide.shapes[0].text = response["deliverables"]
#         slide.shapes[3].text = response["acceptance"]
# 
#         slides.save(pptx_filename)
#         return None
