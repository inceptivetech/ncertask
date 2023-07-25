import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os.path
import configparser
from ratelimit import limits,sleep_and_retry


def create_chunks_from_pdf(dir_path,pdf_path,emb_path):
    pdf_folder_path = dir_path + "/"+ pdf_path + "/class5/"
    pdf_contents= ""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    for pdfile in os.listdir(pdf_folder_path):
        print(pdf_folder_path + "Folder " +pdfile + " File " )
        if pdfile is not None:
            pdf_file_name = os.path.splitext(pdfile)[0]
            embedding_path = dir_path + "/"+ emb_path + "/class5/"
            embedding_file_name = embedding_path+pdf_file_name+".pkl"
            main_embedded_file = embedding_path+"/class5_store.pkl"
            if os.path.exists(main_embedded_file):
                #do nothing
                print("embeddings are present for "+main_embedded_file)
            else:
                pdf_reader = PdfReader(pdf_folder_path+"/"+pdfile)   
                for page in pdf_reader.pages:
                    pdf_text = ""
                    pdf_text += page.extract_text()
                    pdf_contents = pdf_contents +pdf_text
    print("pdf contents "+pdf_contents)
    pdf_chunks =  text_splitter.split_text(text=pdf_contents) 
    #print("pdf contents "+pdf_chunks)       
    create_embeddings(pdf_chunks,main_embedded_file)
    

def create_single_vector_store(root, emb_path):
    vector_path = root+"/"+emb_path+"/class5/"
    class5_store=[]
    for filename in os.listdir(vector_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(vector_path, filename)
            with open(file_path, 'rb') as file:
                vectors = pickle.load(file)
                class5_store.append(vectors)
    output_filename = 'ncert_store.pkl'
    with open(output_filename, 'wb') as file:
        pickle.dump(class5_store, file)
    print("Vector store created and saved as", output_filename)


def create_embeddings(pdf_chunks,emb_file_name):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(pdf_chunks,embedding=embeddings)
    with open(f"{emb_file_name}",'wb') as f:
        pickle.dump(vectorstore,f)
    print("embeddings computed and writtern " +emb_file_name )

def main():
    load_dotenv()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path= dir_path+"\config.ini"
    path_exists = os.path.exists(config_path)
    print("config path: " +config_path)
    config = configparser.ConfigParser()

    if path_exists:
        config.read(config_path)
        file_config = config["FILEPATH"]
        pdf_path = file_config["PDF_PATH"]
        embedding_path = file_config["EMBEDDING_PATH"]
        create_chunks_from_pdf(dir_path,pdf_path,embedding_path)
       # create_single_vector_store(dir_path, embedding_path)
        print("embeddings computed")
    
if __name__ == '__main__':
    main()