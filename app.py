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
import os


with st.sidebar:
    st.title('LLM NCERT APP')
    st.markdown(
        '''
        ### This app LLM Powered
        '''
    )
    add_vertical_space(5)
    st.write('Made by Inceptive')

def get_pdf_text():
    

def main():
    st.header('Chat with NCERT')
    load_dotenv() 
    pdf = st.file_uploader('Upload your pdf', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl",'rb') as f:
                vectorstore = pickle.load(f)
            st.write('Embedding loaded from disk')
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings) 
            with open(f"{store_name}.pkl",'wb') as f:
                pickle.dump(vectorstore, f)
            st.write("embedding computed")
        query = st.text_input("Ask questions about your pdf files:")
        st.write(query)

        if(query):
            docs = vectorstore.similarity_search(query=query, k=2)
            st.write(docs)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents= docs, question=query)
            st.write(response)

        

if __name__ == '__main__':
    main()