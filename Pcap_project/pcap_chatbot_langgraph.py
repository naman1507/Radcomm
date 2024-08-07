from typing import Annotated
import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st

import tempfile
import os
from pprint import pprint
### Router
import gradio as gr
import tempfile
from typing import Literal




#second draft
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import json
import base64


import os  

os.environ["OPENAI_API_KEY"] = "<your-api-key>"



# name of the text file in which pcap will get parsed 
text_output = 'output_pcap_graph.txt'  
  
# Check if the output file exists  
if os.path.exists(text_output):  
    # Delete the file  
    os.remove(text_output)  
    print(f"File '{text_output}' has been deleted.")  
else:  
    print(f"File '{text_output}' does not exist.")  

import os  
import shutil  
  
# Define the folder path in which to store the pcap file   
folder_path = '/home/anshul/Documents/langchain/langgraph/uploaded_files/'  
  
# Check if the folder exists  
if os.path.exists(folder_path):  
    # List all files in the folder  
    files = os.listdir(folder_path)  
      
    # Check if there are any files in the folder  
    if files:  
        # Iterate over each file and delete it  
        for file in files:  
            file_path = os.path.join(folder_path, file)  
            try:  
                if os.path.isfile(file_path) or os.path.islink(file_path):  
                    os.unlink(file_path)  # Remove the file or link  
                elif os.path.isdir(file_path):  
                    shutil.rmtree(file_path)  # Remove the directory and its contents  
                print(f"Deleted: {file_path}")  
            except Exception as e:  
                print(f"Failed to delete {file_path}. Reason: {e}")  
        print("All files deleted successfully.")  
    else:  
        print("No files found in the folder.")  
else:  
    print("The specified folder does not exist.") 

def extract_pcap_text(pcap_file, output_file):
    protocols = "s1ap or ngap or sip or gtpv2 or diameter or http2 or pfcp or lcsap or ulp or dns or icmp or tcp or rtp or f1ap or udp or http"
    command = f"tshark -r {pcap_file} -V -Y '{protocols}' > {output_file}"
    subprocess.run(command, shell=True, check=True)


# Route query class used to get ouput in a specific format 
class RouteQuery(BaseModel):
    """Route a user query to the most relevant node."""

    datasource: Literal["pcap", "chat"] = Field(
        ...,
        description="Given a user question route it . If the query is a general query then route it to chat , but if the query asks for something related to pcap analysis route it to pcap , which will then parse the given pcap file",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a pcap decode / pcap related query or general response.
if there is a mention of the pcap file or just pcap in the query the user is giving reference to the uploaded pcap file
the pcap decode contains a pcap file parser and finds extracts relevant information from it to answer user query for pcap analysis for telecom or internet networks  
the chat on the other is just a basic query responder so that the pcap decode dosent get called for general querys of the user  """
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
# chaining used for the route model 
question_router = route_prompt | structured_llm_router
# print(
#     question_router.invoke(
#         {"question": "summary of the pcap?"}
#     )
# )
# print(question_router.invoke({"question": "what is langrraph ?"}))



from typing import List

from typing_extensions import TypedDict

# define the graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        pcap_path: pcap_path
        hitory: hitory of messages
    """

    question: str
    pcap_path: str
    answer:str
    history:str


#contextualize the user current  query with the last user message and generate a new contextualized user query
def contextualize_query(state):
    print("contextualize_query___")
    contextualize_q_system_prompt = ("""## Prompt: Rewriting User Questions with PCAP Context

**Objective:**  Transform a user's question into a standalone, contextually rich question by incorporating relevant information from the chat history, specifically related to PCAP file analysis.

**Instructions:**
1. **read the user asked query clearly and understand what it wants 
1. **PCAP Contextualization:** If the user's question pertains to a PCAP file, prioritize incorporating relevant details from previous PCAP analysis within the chat history.
2. **History Integration:** If the user's question implicitly or explicitly refers to previous discussions or analyses (especially regarding PCAP files), integrate those relevant details into the reformulated question.  **If no relevant history is found, return the original question as is.**
3. **Standalone Clarity:**  The final output should be a clear, self-contained question that can be understood without needing the chat history.
4. **Reformulation or Passthrough:** If the user's question already meets the above criteria, return it as is. Otherwise, reformulate the question to include the necessary context.
5. **No Answering:**  Do NOT provide an answer to the question. Only reformulate the question as needed.
6. **Avoid File Names:** Do not include specific PCAP file names in the reformulated question. Instead, refer to it generally as "the uploaded PCAP file".
7. If the user query has no connection with the past history or any connection with pcap internet telecom analysis pass it as it is 

**Example:**

**Chat History:**
** 
* User: Can you analyze this PCAP file? [attaches file: `malware_traffic.pcap`]
* Assistant: Analysis of the uploaded PCAP file shows communication with a known malicious IP address.

**User Question:** What type of malware is it?

**Reformulated Question:** Based on the analysis of the uploaded PCAP file, which showed communication with a known malicious IP address, what type of malware is likely involved?

**User Question (Unrelated to PCAP):**  What is the current weather in London?

**Reformulated Question (No changes needed):** What is the current weather in London?


**In summary, given a user question and chat history, generate a standalone question enriched with relevant PCAP analysis context from the history, ensuring clarity and comprehensibility without requiring access to the original conversation.  Remember to refer to any PCAP file generally as "the uploaded PCAP file". If no relevant history is found, return the original question as is.**
   """ )

    history = state["history"]
    if history is None:  
        history = " " 
        print("No history")

    question = state["question"]
    messages = [
    (
        "system",
        contextualize_q_system_prompt,
    ),
    ("human", f"history of conversation {history} current asked query : {question}"),
    ]   
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    query = llm.invoke(messages).content
    return {"question": query,"history": history+query}

    


#route question node
def route_question(state):
    print("route question ----")
    """
    Route question to pcap_decode.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "pcap":
        print("---ROUTE QUESTION TO PCAP---")
        return "pcap"
    elif source.datasource == "chat":
        print("---ROUTE QUESTION TO CHAT---")
        return "chat"
    

def chat_basic(state):
    """
    Generate answer for the basic chat mode

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE- basic-chat---")
    question = state["question"]
        
    history = state["history"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    messages = [
    (
        "system",
        "For the given user query give the best answers ",
    ),
    ("human", question),
    ]     

    output = llm.invoke(messages).content

    
    return {"answer":output,"history": history + "\n" +output}



def pcap_parse(state):
    """
    Use the parsed Pcap file data and find user query related that will be helpful to answer the user question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---pcap---")
    question = state["question"]
    pcap_file = state["pcap_path"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
   
    text_output = 'output_pcap_graph.txt'
    # Check if the output file exists  
    if os.path.exists(text_output):  
        # Check if the file is not empty  
        if os.path.getsize(text_output) > 0:  
            with open(text_output, 'r') as file:  
                content = file.read()  
                print("Reading existing output file:")  
        else:  
            # File is empty, run the extract function  
            print("Output file is empty. Extracting text representation...")  
            extract_pcap_text(pcap_file, text_output)  
            with open(text_output, 'r') as file:  
                content = file.read()  
                print("Reading newly extracted output file:")  
    else:  
        # File does not exist, run the extract function  
        print("Output file does not exist. Extracting text representation...")  
        extract_pcap_text(pcap_file, text_output)  
        with open(text_output, 'r') as file:  
            content = file.read()  
            print("Reading newly extracted output file:")  
    messages = [
    (
        "system",
        "Given the decoded text from a pcap file extract the relevant data from the parsed text from pcap file which will be needed to answer the query  ",
    ),
    ("human", f"decoded pcap file data : {content}  the user query asked {question}"),
    ]   
    output = llm.invoke(messages).content

    
    return {"answer": output}


def chat_pcap(state):
    """
    Generate final answer using the contextulized user query and the pcap extracted data 

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE-pcap-query--")
    question = state["question"]
    answer = state["answer"]
        
    history = state["history"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    messages = [
    (
        "system",
        "You are an expert in internet and telecom network analysis and troubleshooting. You are proficient in analyzing PCAP files and understanding network protocols. Your task is to help users understand the information contained within PCAP files and answer their questions related to network behavior and potential issues. ",
    ),
    ("human", f"Here is the realvant information from the parsed pcap file {answer} and here is the question ansked by the user {question}"),
    ]     

    output = llm.invoke(messages).content

    
    return {"answer":output,"history": history + "\n" +output}




from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver

#memory that saves the state of each graph flow 
memory = SqliteSaver.from_conn_string(":memory:")

workflow = StateGraph(GraphState)
workflow.add_node("contextualize_query", contextualize_query)
workflow.add_node("basic_chat", chat_basic)
workflow.add_node("pcap_chat", chat_pcap)
workflow.add_node("pcap_parse", pcap_parse)
workflow.add_edge(START, "contextualize_query")

workflow.add_conditional_edges(
    "contextualize_query",
    route_question,
    {
        "pcap": "pcap_parse",
        "chat": "basic_chat",
    },
)
workflow.add_edge("basic_chat", END)
workflow.add_edge("pcap_parse","pcap_chat")
workflow.add_edge("pcap_chat", END)
app = workflow.compile(checkpointer=memory)

import gradio as gr
import shutil

import tempfile

# function made for gradio to get user response from the langgraph workflow
def process_query_and_pcap(query):
        file_path = get_file_path()
        
        
        config = {"configurable": {"thread_id": "1"}}  
        inputs = {
        "question": query,
        "pcap_path": file_path
        }
        # The config is the **second positional argument** to stream() or invoke()!
        events = app.stream(
            inputs, config , stream_mode="values"
        )
        for event in events:
            if "answer" in event:
                final = event["answer"]
        response = event["answer"]


        return response
import os  

# simple function to get the path of the pcap file  
def get_file_path():  
    # Path of the folder where the pcap file is stored
    folder_path = '/home/anshul/Documents/langchain/langgraph/uploaded_files/'  
      
    # Initialize the file_path variable  
    file_path = None  
      
    # Check if the folder exists  
    if os.path.exists(folder_path):  
        # List all files in the folder  
        files = os.listdir(folder_path)  
          
        # Check if there are any files in the folder  
        if files:  
            # Get the path of the first (and only) file found  
            first_file = files[0]  
            file_path = os.path.join(folder_path, first_file)  
            print(f"File found: {file_path}")  
        else:  
            print("No files found in the folder.")  
    else:  
        print("The specified folder does not exist.")  
      
    # Return the file_path variable  
    return file_path  
  

#simple function for gradio to store the pcap file given by the user 
def upload_file(file):
    #path of folder to upload the pcap file 
    upload_folder = "/home/anshul/Documents/langchain/langgraph/uploaded_files/"
    if not os.path.exists(upload_folder):
        os.mkdir(upload_folder)
    shutil.copy(file,upload_folder )
    gr.Info("  .pcap file uploaded successfully!")



# gradio code for the user interface 
with gr.Blocks() as demo:
    gr.Markdown("A chatbot for pcap analysis.")
    with gr.Row():
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot()
    msg = gr.Textbox()
    upload_button = gr.UploadButton("Click to upload file")
    clear = gr.Button("Clear")
    upload_button.upload(upload_file, upload_button)
    
    def respond(message, chat_history):
        # Get the last user message
        query = message
        # Get the file path from the file component
        # file_path = file.value.name if file.value is not None else None

        # Process the query and pcap file
        response = process_query_and_pcap(query)

        # Update the chat history
        chat_history.append((message, response))
        return "", chat_history
    
    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()