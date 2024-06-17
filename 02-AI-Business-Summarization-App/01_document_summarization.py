# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# FIRST AI-POWERED BUSINESS APP: PART 1
# ***
# GOAL: Exposure to using LLM's, Document Loaders, and Prompts

# IMPORTANT: 
# 1. LEARNING WITH A FAST TRACK PROJECT IS THE BEST WAY TO GET STARTED
# 2. BECAUSE WE ARE DIVING IN, SOME OF THIS MAY FEEL UNCOMFORTABLE
# 3. I WILL FILL IN THE GAPS LATER ON IN THE COURSE

# LIBRARIES AND SETUP

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

import yaml
from pprint import pprint

OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# 1.0 SIMPLE SUMMARIZER

# * DOCUMENT LOADER
loader = PyPDFLoader("02-AI-Business-Summarization-App/pdf/NIKE-Inc-Q3FY24-OFFICIAL-Transcript-FINAL.pdf")

docs = loader.load()

## 1 for each page
len(docs)

# * LLM MODEL
model = ChatOpenAI(
    api_key = OPENAI_API_KEY,
    model = "gpt-3.5-turbo",
    temperature = 0
)

# * LLM CHAINS

summarizer_chain = load_summarize_chain(
    llm = model,
    chain_type = "stuff"
)

pprint(dict(summarizer_chain))

response = summarizer_chain.invoke(docs)

pprint(response['output_text'])

# 2.0 EXPANDING WITH PROMPT TEMPLATES

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

pprint(response["output_text"])

# CONCLUSIONS:
# 1. THIS IS A VERY SIMPLISTIC EXAMPLE, BUT ALREADY YOU CAN SEE THE POWER
# 2. WHAT CHANGES CAN YOU MAKE TO THE PROMPT TEMPLATE TO MODIFY THE OUTPUT? (e.g. "Produce a table" , "Write 1 page report with the following headings and content sections...")
# 3. THIS IS JUST THE START
# 4. NEXT WE NEED TO AUTOMATE IT


