#import streamlit as st
import time
from dotenv import load_dotenv
load_dotenv()
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import p
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os.path
import configparser
from supabase import create_client
import openai

supa_url = os.environ.get("SUPABASE_URL")
supa_key = os.environ.get("SUPABASE_KEY")
api_key = os.environ.get("OPENAI_API_KEY")
supabase = create_client(supabase_url=supa_url, supabase_key=supa_key)
print("client connected")

def create_chunks_embeddings(pdf_folder_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 200,
        length_function = len
    )
    for pdfile in os.listdir(pdf_folder_path):
        print(pdf_folder_path + "Folder " +pdfile + " File " ) 
        if pdfile is not None:
            #populate_class_page(pdfile)
            pdf_contents=""
            pdf_file_name = os.path.splitext(pdfile)[0]
            pdf_reader = PdfReader(pdf_folder_path+"/"+pdfile)   
            for page in pdf_reader.pages:
                pdf_text = ""
                pdf_text += page.extract_text()
                pdf_contents = pdf_contents +pdf_text
                #print("pdf contents "+pdf_contents)
                pdf_chunks =  text_splitter.split_text(text=pdf_contents) 
           # print("pdf contents of "+pdf_file_name)  
           # print(pdf_chunks)
            populate_class_page_embeddings(pdf_chunks,pdf_file_name)          

def populate_class_page(file_obj):
    pdf_file_name = os.path.splitext(file_obj)[0]
    data = supabase.table("class_file").insert([{
        "class_name":"Class5",
        "file_path":pdf_file_name,
        "active":True,
        "embeddings":False
    }]).execute()
    print(pdf_file_name +" inserted")


def populate_class_page_embeddings(pdf_chunks,file_name):
    #TODO
    """
        1. Define AI model to use
        2. text-embeddings-ada-002
        3. create embeddings
        4. update content_chunk
        5. update classpage_embeddings
        6. push the json to embedding_obj
        7. push embedding_obj to supabase
        8. test the similarity search
    """
    embedding_obj = []
    embedding_json_dict = {
        "class_name":"Class5",
        "file_path":file_name,
        "active":True,
        "content_chunk":"",
        "classpage_embeddings":"",
    }
    #ai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=api_key)
 
    openai.api_key = api_key
    num_chunks_per_min = 3
    print(f"pdf chunk length: {len(pdf_chunks)}")
    time.sleep(60)
    #below implementation needs to be replaced with creating multiple insert objects for each pdf to improve performance
    for i in range(0, len(pdf_chunks),num_chunks_per_min):
        chunk_batch = pdf_chunks[i:i+num_chunks_per_min]
        for chunk in chunk_batch:
            payload = {
                "model":"text-embedding-ada-002",
                "input":chunk.replace('[{', '').replace('}]', '')
                }
            sanchunk = chunk.replace('[{', '').replace('}]', '')
            response = openai.Embedding.create(model="text-embedding-ada-002",input=sanchunk)

            adaembeddings = response['data'][0]["embedding"]
            #print(response)
            #print(response['data'][0]["embedding"])
            data = supabase.table("class_page_embeddings").insert([{
                "class_name":"Class5",
                "file_path":file_name,
                "active":True,
                "content_chunk": sanchunk,
                "classpage_embeddings":adaembeddings
            }]).execute()
        if i + num_chunks_per_min < len(pdf_chunks):
            print("sleeping for a min")
            time.sleep(60)
    print("All chunks completed")

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path= dir_path+"\config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    file_config = config["FILEPATH"]
    pdf_file_path = dir_path + "/" + file_config["PDF_PATH"]+"/class5/"
    create_chunks_embeddings(pdf_file_path)

if __name__ == '__main__':
    main()