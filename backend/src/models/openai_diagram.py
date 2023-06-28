# """OpenAI API to generate a software architecture.
# 
# Generative AI based on OpenAI's ChatGPT to create a software
# architectures based on a single prompt.
# """
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
# from diagrams import Diagram, Edge, Cluster
# from diagrams.c4 import (
#     Person,
#     Container,
#     Database,
#     System,
#     SystemBoundary,
#     Relationship,
# )
# 
# 
# # Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
# sys.path.append(os.path.abspath("../../../"))
# 
# from src.models import ChatBot
# 
# 
# class OpenAIDiagram(ChatBot):
#     """ChatGPT agent for generating software architecture diagrams."""
# 
#     title = ""
#     architecture = 0
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
#             self.identity = (
#                 "You are a software architect for a company called Capgemini."
#             )
#         else:
#             self.identity = self.config["prompt"]["identity"]
#         if not self.config.get("prompt", {}).get("intent", False):
#             self.intent = "You draw software architecture diagrams for a software project based on the input prompt."
#         else:
#             self.intent = self.config["prompt"]["intent"]
#         if not self.config.get("prompt", {}).get("behavior", False):
#             self.behavior = "You are concise, knowledgeable, and intelligent."
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
#         try:
#             self.architecture = int(response["text"])
#         except:
#             self.architecture = 0
# 
#         if self.architecture == 0:
#             self.prompt = self._mvc_prompt()
#         elif self.architecture == 1:
#             self.prompt = self._layered_prompt()
#         else:
#             self.prompt = self._mvc_prompt()
# 
#         chain = self._chain()
#         response = chain(user_input)
#         output = self._text_json_to_dict(response["text"])
#         self.title = output["title"]
# 
#         self.prompt = self._prompt()  # reset to default for future use
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
#         prompt = PromptTemplate(
#             template=f"Based on the software application described in the input prompt, say 1 if a layered n-tiered architecture pattern with presentation, application, persistence, and database layers is most appropriate. Else, say 0 if a model-view-controller architecture pattern is most appropriate. Do not say anything else after that.\n----------------\n{{question}}\n",
#             input_variables=["question"],
#         )
# 
#         return prompt
# 
#     def _mvc_prompt(self):
#         """Private method to get a MVC prompt.
# 
#         Args:
#             None
#         Returns:
#             prompt (obj): LangChain prompt template object
#         """
# 
#         response_schemas = [
#             ResponseSchema(
#                 name="architecture",
#                 description="Based on the software application described in the input prompt, say 1 if a layered n-tiered architecture pattern with presentation, application, persistence, and database layers is most appropriate. Else, say 0 if a model-view-controller architecture pattern is most appropriate. Do not say anything else after that.",
#             ),
#             ResponseSchema(
#                 name="title",
#                 description="Write a short title of a software application based on the input prompt in less than 10 words.",
#             ),
#             ResponseSchema(
#                 name="user_desc",
#                 description="Write a description describing the intended end user of a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="model_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'model' component of a model-view-controller architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="model_desc",
#                 description="Write a description describing the 'model' component of a model-view-controller architecture for a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="view_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'view' component of a model-view-controller architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="view_desc",
#                 description="Write a description describing the 'view' component of a model-view-controller architecture for a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="controller_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'controller' component of a model-view-controller architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="controller_desc",
#                 description="Write a description describing the 'controller' component of a model-view-controller architecture for a software application based on the input prompt with approximately 10 to 15 words.",
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
#     def _layered_prompt(self):
#         """Private method to get a MVC prompt.
# 
#         Args:
#             None
#         Returns:
#             prompt (obj): LangChain prompt template object
#         """
# 
#         response_schemas = [
#             ResponseSchema(
#                 name="architecture",
#                 description="Based on the software application described in the input prompt, say 1 if a layered n-tiered architecture pattern with presentation, application, persistence, and database layers is most appropriate. Else, say 0 if a model-view-controller architecture pattern is most appropriate. Do not say anything else after that.",
#             ),
#             ResponseSchema(
#                 name="title",
#                 description="Write a short title of a software application based on the input prompt in less than 10 words.",
#             ),
#             ResponseSchema(
#                 name="user_desc",
#                 description="Write a description describing the intended end user of a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="presentation_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'presentation' layer of a layered presentation-application-persistence-database architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="presentation_desc",
#                 description="Write a description describing the 'presentation' layer of a layered presentation-application-persistence-database architecture for a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="application_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'application' layer of a layered presentation-application-persistence-database architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="application_desc",
#                 description="Write a description describing the 'application' layer of a layered presentation-application-persistence-database architecture for a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="persistence_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'persistence' layer of a layered presentation-application-persistence-database architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="persistence_desc",
#                 description="Write a description describing the 'persistence' layer of a layered presentation-application-persistence-database architecture for a software application based on the input prompt with approximately 10 to 15 words.",
#             ),
#             ResponseSchema(
#                 name="database_tech",
#                 description="List less than 4 technologies separated by commas that would be used in the 'database' layer of a layered presentation-application-persistence-database architecture for an software application based on the user input.",
#             ),
#             ResponseSchema(
#                 name="database_desc",
#                 description="Write a description describing the 'database' layer of a layered presentation-application-persistence-database architecture for a software application based on the input prompt with approximately 10 to 15 words.",
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
#     def generate_diagram(self, response, png_filename):
#         """Generate a software architecture diagram based on response.
# 
#         Args:
#             response (dict): Formatted JSON response
#             png_filename (str): Path to output image
#         Returns:
#             None
#         """
# 
#         if self.architecture == 0:
#             self._generate_mvc_diagram(response, png_filename)
#         elif self.architecture == 1:
#             self._generate_layered_diagram(response, png_filename)
#         else:
#             logging.error("The architecture has not been defined yet.")
#             sys.exit()
# 
#     def _generate_mvc_diagram(self, response, png_filename):
#         """Private method to generate a MVC architecture diagram.
# 
#         Args:
#             response (dict): Formatted JSON response
#             png_filename (str): Path to output image
#         Returns:
#             None
#         """
# 
#         png_filename = re.sub(".png$", "", png_filename)
#         graph_attr = {
#             "pad": "0",
#             "spline": "spline",
#             "fontname": "Arial",
#         }
#         node_attr = {
#             "fontname": "Arial",
#         }
# 
#         with Diagram(
#             response["title"],
#             show=False,
#             graph_attr=graph_attr,
#             node_attr=node_attr,
#             filename=png_filename,
#             direction="LR",
#         ):
#             user = Person(
#                 name="User",
#                 description=response["user_desc"],
#             )
#             with Cluster("MVC Architecture"):
#                 model = Container(
#                     name="Model",
#                     technology=response["model_tech"],
#                     description=response["model_desc"],
#                 )
#                 view = Container(
#                     name="View",
#                     technology=response["view_tech"],
#                     description=response["view_desc"],
#                 )
#                 controller = Container(
#                     name="Controller",
#                     technology=response["controller_tech"],
#                     description=response["controller_desc"],
#                 )
# 
#             view >> Relationship("") >> user
#             controller << Relationship("") << user
#             user - Relationship(style="invis") - model
# 
#             model >> controller
#             model << controller
#             controller >> view
# 
#         return None
# 
#     def _generate_layered_diagram(self, response, png_filename):
#         """Private method to generate a layered n-tier architecture diagram.
# 
#         Args:
#             response (dict): Formatted JSON response
#             png_filename (str): Path to output image
#         Returns:
#             None
#         """
# 
#         png_filename = re.sub(".png$", "", png_filename)
#         graph_attr = {
#             "pad": "0",
#             "spline": "spline",
#             "fontname": "Arial",
#         }
#         node_attr = {
#             "fontname": "Arial",
#         }
# 
#         with Diagram(
#             response["title"],
#             show=False,
#             graph_attr=graph_attr,
#             node_attr=node_attr,
#             filename=png_filename,
#             direction="LR",
#         ):
#             user = Person(
#                 name="User",
#                 description=response["user_desc"],
#             )
#             with Cluster("Layered n-tier Architecture"):
#                 presentation = Container(
#                     name="Presentation Layer",
#                     technology=response["presentation_tech"],
#                     description=response["presentation_desc"],
#                 )
#                 application = Container(
#                     name="Application Layer",
#                     technology=response["application_tech"],
#                     description=response["application_desc"],
#                 )
#                 persistence = Container(
#                     name="Persistence Layer",
#                     technology=response["persistence_tech"],
#                     description=response["persistence_desc"],
#                 )
#                 database = Database(
#                     name="Database Layer",
#                     technology=response["database_tech"],
#                     description=response["database_desc"],
#                 )
# 
#             (
#                 user
#                 >> Relationship()
#                 >> presentation
#                 >> application
#                 >> persistence
#                 >> database
#             )
#             (
#                 user
#                 << Relationship()
#                 << presentation
#                 << application
#                 << persistence
#                 << database
#             )
# 
#         return None
# 
#     def _text_json_to_dict(self, text):
#         """Private method to convert text json to a dict.
# 
#         Args:
#             text (str): String with json text
#         Returns:
#             output (dict): Converted text to dict
#         """
# 
#         text = str(text)
#         text = re.sub("\n", "", text)
#         text = re.sub("^```json", "", text)
#         text = re.sub("```.*$", "", text)
#         output = json.loads(text)
# 
#         return output
