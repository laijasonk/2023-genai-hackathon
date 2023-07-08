# """OpenAI API to generate a chatbot with contextual information.
#
# Chatbot based on OpenAI's ChatGPT with input context embeddings based on
# documents stored in a FAISS database.
# """
#
# import os
# import sys
# import logging
#
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain import PromptTemplate
# from langchain.prompts.chat import SystemMessagePromptTemplate
# from langchain.vectorstores import FAISS
#
# # Path must be defined (e.g. PYTHONPATH="/path/to/repo/backend")
# sys.path.append(os.path.abspath("../../../"))
#
# from src.models import ChatBot
#
#
# class OpenAIContext(ChatBot):
#     """ChatGPT agent with context."""
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
#             self.identity = "You are an AI assistent for our company called Capgemini. Try to format the response and make it ready for html. When it is needed use html tags like <br>, <p>, <b>, <strong>, <i>, <em>, <mark>, <small>. When you are using <br>, use double of that to make it nicer."
#         else:
#             self.identity = self.config["prompt"]["identity"]
#         if not self.config.get("prompt", {}).get("intent", False):
#             self.intent = "You are answering questions in a conversation with an employee.\nUse the context provided below to answer the users question."
#         else:
#             self.intent = self.config["prompt"]["intent"]
#         if not self.config.get("prompt", {}).get("behavior", False):
#             self.behavior = "You are conversational, helpful, and friendly.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer."
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
#         config_faiss = self.config.get("faiss", {})
#         persist_directory = config_faiss.get("persist_directory", "./data/faiss")
#         index_names = config_faiss.get("index_names", ["example"])
#
#         vectorstore = FAISS.from_texts([""], embedding=self.embeddings)
#         for local_fn in index_names:
#             db = FAISS.load_local(
#                 persist_directory,
#                 index_name=local_fn,
#                 embeddings=self.embeddings,
#             )
#             vectorstore.merge_from(db)
#
#         retriever = vectorstore.as_retriever(
#             # search_type="similarity",
#             # search_kwargs={"k": 1, "include_metadata": True}
#         )
#
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             memory=self.memory,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True,
#             get_chat_history=lambda h: h,
#             verbose=False,
#         )
#
#         # Hack because of bug in LangChain where prompt argument does nothing
#         chain.combine_docs_chain.llm_chain.prompt.messages[
#             0
#         ] = SystemMessagePromptTemplate(prompt=self.prompt)
#
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
#             output_key="answer",
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
#         template = f"{self.identity}\n{self.intent}\n{self.behavior}\n----------------\nContext:\n{{context}}\n----------------\n"
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
#             input_variables=["context"],
#             template=self.template,
#             output_parser=None,
#             partial_variables={},
#         )
#
#         return prompt
