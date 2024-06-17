# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# FIRST AI-POWERED BUSINESS APP: PART 2
# ***
# GOAL: Exposure to using LLM's, Document Loaders, and Prompts

# streamlit run 02-AI-Business-Summarization-App/02_document_summarizer_app.py


import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain

import streamlit as st
import subprocess
import shutil
import os
from tempfile import NamedTemporaryFile

# Load API Key
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']
MODEL = "gpt-3.5-turbo"

# generate PDF function
def generate_pdf_with_quarto(markdown_text):
    with NamedTemporaryFile(delete = False, suffix = ".qmd", mode = "w") as md_file:
        md_file.write(markdown_text)
        markdown_path = md_file.name
    
    pdf_file_path = markdown_path.replace(".qmd", ".pdf")
    
    # use quarto command line instead of python for more complex rendering
    subprocess.run(["quarto", "render", markdown_path, "--to", "pdf"], check = True)
    
    os.remove(markdown_path)
    return pdf_file_path
    
# generate move pdf to downloads function
def move_file_to_downloads(pdf_file_path):
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    destinations_path = os.path.join(downloads_path, os.path.basename(pdf_file_path))
    shutil.move(pdf_file_path, destinations_path)
    return destinations_path

# 1.0 LOAD AND SUMMARIZE FUNCTION
def load_and_summarize(file):
    
    with NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
        
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        model = ChatOpenAI(
            api_key = OPENAI_API_KEY,
            model = "gpt-3.5-turbo",
            temperature = 0
        )
    
        prompt_template = """
        Write a business report from the following earnings call transcript:
        {text}

        Use the following Markdown format:
        # Insert Descriptive Report Title

        ## Earnings Call Summary
        Use 3 to 7 numbered bullet points

        ## Important Financials
        Describe the most important financials discussed during the call. Use 3 to 5 numbered bullet points.

        ## Key Business Risks
        Describe any key business risks discussed on the call. Use 3 to 5 numbered bullets.

        ## Conclusions
        Conclude with any overaching business actions that the company is pursuing that may have a positive or negative implications and what those implications are. 
        """

        prompt = PromptTemplate(
            template = prompt_template,
            input_variables = ["text"]
        )

        llm_chain = LLMChain(prompt = prompt, llm = model)

        stuff_chain = StuffDocumentsChain(llm_chain = llm_chain, document_variable_name = "text")

        response = stuff_chain.invoke(docs)
        
    finally:
        os.remove(file_path)
        
    return response['output_text']

# 2.0 STREAMLIT INTERFACE
st.set_page_config(page_title = "Earnings Call Transcript Summarizer", layout = "wide", page_icon = "📄")
st.title("PDF Earnings Call Summarizer")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload the PDF Document:")
    uploaded_file = st.file_uploader("Choose a file", type = "pdf")
    if uploaded_file:
        summarize_flag = st.button("Summarize Document", key = "summarize_button")
        
    if uploaded_file and summarize_flag:
        with col2:
            with st.spinner("Summarizing Document..."):
                summary = load_and_summarize(uploaded_file)
                st.subheader("Summarizer Results:")
                st.markdown(summary)
                
                pdf_file = generate_pdf_with_quarto(summary)
                download_path = move_file_to_downloads(pdf_file)
                st.markdown(f"PDF downloaded to your Downloads folder: {download_path}.")
                
        
    else:
        with col2:
            st.write("No file uploaded. Please upload a PDF file to summarize.")

# CONCLUSIONS:
#  1. WE CAN SEE HOW APPLICATIONS LIKE STREAMLIT ARE A NATURAL INTERFACE TO AUTOMATING THE LLM TASKS
#  2. BUT WE CAN DO MORE. 
#     - WHAT IF WE HAD A FULL DIRECTORY OF PDF'S?
#     - WHAT IF WE WANTED TO DO MORE COMPLEX ANALYSIS?
