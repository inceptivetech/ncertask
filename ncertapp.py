import configparser
import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from annoy import AnnoyIndex
import os
import torch

# Function to retrieve vectors from the loaded stores based on similarity search
def retrieve_vector(input_text, vector_stores, index):
    vectors = []
    for vector_store in vector_stores:
        vector = vector_store.get(input_text, None)
        if vector is not None:
            vectors.append(vector)
    
    if vectors:
        vectors = torch.cat(vectors, dim=-1).numpy()
        indices, _ = index.get_nns_by_vector(vectors, 1, include_distances=False)
        return torch.from_numpy(vectors[indices[0]])
    else:
        return torch.zeros(1, 768)  # Default vector of zeros

# Function to generate response using question-answering system
def generate_response(query, model, ncert_vector, index):
    vectors = retrieve_vector(query, ncert_vector, index)
    input_text = "generate: " + query
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = torch.cat([input_ids, vectors], dim=-1)
    
    output = model.generate(input_ids)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

def main():
    st.header('Chat with NCERT')
    load_dotenv() 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path= dir_path + "/config.ini"

    print("config path: " + config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    file_config = config["FILEPATH"]
    embedding_path = dir_path + "/" + file_config["EMBEDDING_PATH"] + "/class5/"

    # Load vector stores
    ncert_vector = []
    if os.path.exists(embedding_path):
        for vec_file in os.listdir(embedding_path):
            with open(os.path.join(embedding_path, vec_file), 'rb') as f:
                vectorstore = pickle.load(f)
                ncert_vector.append(vectorstore)
    st.write('ncert vector created and loaded from disk')

    query = st.text_input("Ask questions about your pdf files:")
    st.write(query)

    if query:
        response = generate_response(query, model, ncert_vector, index=index)
        st.write(response)

if __name__ == '__main__':
    # Initialize the approximate nearest neighbor index
    vector_dimension = 768  # Modify the dimension according to your vector representation
    index = AnnoyIndex(vector_dimension, 'euclidean')

    # Load the language model
    model_name = 'gpt2-medium'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Specify the number of trees for the index
    n_trees = 100  # Modify the number of trees according to your preference

    # Perform similarity search setup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = dir_path + "/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    file_config = config["FILEPATH"]
    embedding_path = dir_path + "/" + file_config["EMBEDDING_PATH"] + "/class5/"

    if os.path.exists(embedding_path):
        for vec_file in os.listdir(embedding_path):
            with open(os.path.join(embedding_path, vec_file), 'rb') as f:
                vectorstore = pickle.load(f)
                for key, vector in vectorstore.items():
                    index.add_item(len(ncert_vector), vector.numpy())
                    ncert_vector.append(vector)

    index.build(n_trees)

    main()
