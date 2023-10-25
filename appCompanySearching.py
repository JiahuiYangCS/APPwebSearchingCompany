import gradio as gr
import os
import tempfile
from summarizer_module import OpenAISummarizer
#from PDFtoTXT_module import pdf_to_txt_file

from datetime import datetime
import openai
from pymongo import MongoClient
from langchain.retrievers.web_research import WebResearchRetriever

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import TokenTextSplitter

import logging

#from mongodb_module import MongoDBManager

# connecting to MangoDB

client = MongoClient('localhost', 27017)

db = client['your_database_name']

ccollectionCompany = db['CompanyInformation']




def save_company_in_DB(output_text, api_key):
    
    openai.api_key = api_key
    
    job_description = output_text

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts specific details from company descriptions and output in Chinese."
        },
        {
            "role": "user",
            "content": f"""
            From the provided company description, extract the following details:

            公司名称： {{名称}} 
            公司情况介绍： {{介绍}}
            商业价值:{{商业价值}}
            技术优势:{{技术优势}}
            公司的位置:{{位置}}
            公司的规模:{{规模}}
            商业模式:{{商业模式}}
            市值/融资情况:{{市值/融资}}
            产品:{{产品}}
            专利:{{专利}}
            
            

            {job_description}
            """
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=4000  # output limit, adjustable 
    )

    
    JDoutput_text = response.choices[0]['message']['content'].strip().split("\n")

    parsed_data = {line.split(": ")[0]: line.split(": ")[1] for line in JDoutput_text}

    job_data = {
        "_id": parsed_data["公司名称"],
        "介绍": parsed_data["公司情况介绍"].split('; '),
        "商业价值": parsed_data["商业价值"].split('; '),
        "技术优势": parsed_data["技术优势"],
        "位置": parsed_data["公司的位置"],
        "规模": parsed_data["公司的规模"],
        "商业模式": parsed_data["商业模式"],
        "市值/融资": parsed_data["市值/融资情况"],
        "产品": parsed_data["产品"],
        "专利": parsed_data["专利"]
        
    }

    collectionCompany.insert_one(job_data)
    
    successfulText = f"Company description for {parsed_data['公司名称']} has been stored in MongoDB!"
    dataName = parsed_data['公司名称']
    
    return successfulText, dataName



def saveInMangodb(output_text, api_key):
    return save_company_in_DB(output_text, api_key)
    
    
    
def search(api_key, googleAPI, engineID, task, typed_text=None):
    
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ["GOOGLE_CSE_ID"] = engineID
    os.environ["GOOGLE_API_KEY"] = googleAPI
    
    # Vectorstore
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

    # LLM
    llm = ChatOpenAI(temperature=0)
    
    
    
    search = GoogleSearchAPIWrapper()
    
    text_splitter = TokenTextSplitter(
        chunk_size=4000, chunk_overlap=50
        )
    
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
        text_splitter=text_splitter,
        num_search_results = 3,
        )
    
    company_name = typed_text
    
    user_input = f"give the information about company: {company_name}, include:Location of the company, Size of the company,Business model,Technical advantages,Market value / Financing situation,Products,Patents"
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    docs = web_research_retriever.get_relevant_documents(user_input)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(temp_file.name, 'w') as f:
        for item in docs:
            f.write(f"{item}\n")
    txt_path = temp_file.name
   
    summarizer = OpenAISummarizer(api_key)
    
    if task == "Generate English company information from google searching":
        result = summarizer.summarizeCompanyEN(txt_path)
    elif task == "Generate Chinese company information from google searching":
        result = summarizer.summarizeCompanyCN(txt_path)
    
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Save the result to another temporary file
    summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(summary_file.name, 'w', encoding='utf-8') as f:
        f.write(result)


    return result, summary_file.name

TASK_NAMES = [
    "Generate English company information from google searching",
    "Generate Chinese company information from google searching",
]



DESCRIPTION = """# Company Information Summary from Google Searching
Powered by ChatGPT3.5 and Google search engine

api_key: openAI api key



Using ChatGPT 3.5 Turbo 16k large language model

Using the Langchain freamwork with map and reduce summarization and web research & retriever methods

Map and reduce prompt words can be modified in the summarize function of the summarizer_module.py file as needed

Input type: window typing in

"""
css = """
h1 {
  text-align: center;
}

.contain {
  max-width: 730px;
  margin: auto;
  padding-top: 1.5rem;
}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Group():
        task = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0]
        )
    """with gr.Column():
        inputType = gr.Dropdown(
            label="INPUT_TYPE",
            choices=INPUT_TYPE,
            value="PDF/text file",
            visible=True
        )
        """
    api_key = gr.Textbox(label="openai api key", visible=True)
    
    googleAPI = gr.Textbox(label="google search engine api key", visible=True)
    engineID = gr.Textbox(label="google search engine ID", visible=True)
    
    # file = gr.inputs.File(label="Upload PDF or TXT",optional=True)
        
    typed_text = gr.Textbox(placeholder="Type the company name here...", label="Typed Text", optional=True, visible=True)
        
    btn = gr.Button("Search and Generate")
        
    with gr.Column():
        output_text = gr.Textbox(label="Company Information")

    with gr.Column():
        output_file = gr.outputs.File(label="Download Summary")# Allow users to download the output
    
    btnDB = gr.Button("Save the result/data to data base(Mangodb)")
    
    with gr.Column():
        successfulText = gr.Textbox(label="Storage Success Indicator")
        dataName = gr.Textbox(label="data/record named as: ")
    
    btn.click(
        fn=search,
        inputs=[
            api_key, googleAPI, engineID, task, typed_text,
        ],
        outputs=[output_text, output_file],
        api_name="run",
    )
    
    
    btnDB.click(
        fn=saveInMangodb,  
        inputs=[
            output_text, api_key
        ],
        outputs=[successfulText, dataName],
        api_name="run",
    )

    if __name__ == "__main__":
    demo.queue().launch()