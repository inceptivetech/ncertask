import configparser
import json
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
from langchain.callbacks import get_openai_callback;
from transformers import Text2TextGenerationPipeline, T5Tokenizer
from transformers import Text2TextGenerationPipeline
from annoy import AnnoyIndex
import os
from supabase import create_client
import numpy as np
import requests
import openai


with st.sidebar:
    st.title('LLM NCERT APP')
    st.markdown(
        '''
        ### This app LLM Powered
        '''
    )
    add_vertical_space(5)
    st.write('Made by Inceptive')

def main():
    st.header('Chat with NCERT')
    load_dotenv() 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path= dir_path+"\config.ini"
    
    print("config path: " +config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    file_config = config["FILEPATH"]
    embedding_path = dir_path + "/" + file_config["EMBEDDING_PATH"]+"/class5/"
    query = st.text_input("Ask questions about your pdf files:")
    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_KEY")
    api_key = os.environ.get("OPENAI_API_KEY")
    supabase = create_client(supabase_url=supa_url, supabase_key=supa_key)
    supa_headers = {
        "apikey":supa_key
    }

    st.write(query)
    if(query):
        #generate the query embeddings
        """
        Create the query embeddings 
        Do the similarity search
        Take the top3 search contents and along with query send it to Open AI
        """
        openai.api_key = api_key
        query = query.replace('[{', '').replace('}]', '')
        supa_url = os.environ.get("SUPABASE_URL")
        supa_key = os.environ.get("SUPABASE_KEY")
        supa_api_url = "{supa_url_live}/rpc/{schema}.{function_name}"

        supabase = create_client(supabase_url=supa_url, supabase_key=supa_key)
        

        print("client connected")
        response = openai.Embedding.create(model="text-embedding-ada-002",input=query)
        query_embeddings = response['data'][0]["embedding"]
        #Do the similarity search call the supabase function
        similarity_threshold = 0.6
        matches = 4
        payload = {
            "query_embedding": query_embeddings,
            "similarity_threshould": similarity_threshold,
             "match_count": matches
        }
        json_payload = json.dumps(payload)
        #response = requests.post(supa_url+"/rpc/public.search_ncert",
         #                        headers=supa_headers,data=json_payload)
        #data = response.json()
        #print(data)
        response = supabase.rpc("search_ncert",payload).execute()
        #print(response)
        #retuned_chunks = [item['content_chunk'] for item in response.data]
        content_chunks_string = "\n\n".join(item['content_chunk'] for item in response.data)
        #st.write(content_chunks_string)
        #prepare the data for openai request
        prompt = "Use the following information to answer the query: "+query+content_chunks_string
        print(prompt)
        request_data = {
            "prompt": prompt,
            "apiKey": api_key
        }
        ai_header = {
            "Content_Type":"application/json",
            "Authorization":"Bearer " +api_key
        }
        ai_data = {
        "model": "text-davinci-003",  # For OpenAI API v1, you can use "text-davinci-002" as an equivalent to "OpenAIModel.DAVINCI_TURBO"
        "max_tokens": 250,
        "temperature": 0.0,
        "stream": False
        }

       # airesponse = requests.post('https://api.openai.com/v1/completions',headers=ai_header,json=ai_data)
        openai.api_key = api_key
        airesponse = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0
       )

        st.write(airesponse.choices[0].text)

if __name__ == '__main__':
    main()