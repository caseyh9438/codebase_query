from django.shortcuts import render
import requests
import json
import subprocess
import os 
import pdb
from tqdm import tqdm



def process_repo_util(url: str, model: object):

    """ 
    THIS FUNCTION TAKES IN A REPO URL, CLONES THE REPO TO THE REPO FOLDER, 
    CRAWLS ALL OF THE FILES AND RETUNS AND LIST OF ALL CODE IN THE REPO 
    """

    # assert is_valid_url(url = url) is True, "This URL is not valid."


    """ CHECK IF URL IS ALREADY IN DB, IF SO RETURN KNOWN CODEBASE """
    code_list = model.objects.filter(url = url)

    if code_list.exists():

        code_list = model.objects.get(url = url)
        # import pdb
        #3pdb.set_trace()
        codebase = code_list.items

    else:

        """ PATH TO THE REPO DIR """
        repo_dir =  os.path.join(os.getcwd(), 'repo')

        """ CLONE THE REPO TO THE REPO DIR """
        subprocess.run(["git", "clone", url, repo_dir])

        """ THIS CODEBASE """
        codebase = []

        """ LOOP THROUGH FILE AND APPEND TO CODEBASE LIST """
        for root, dirs, files in os.walk(repo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        file_content = f.read()
                        codebase.append(file_content)
                    except UnicodeDecodeError as error:
                        print(error)

        """ NOW SAVE TO DB """
        code_list = model(url = url)
        code_list.set_items(codebase)
        code_list.save()

    return codebase


def is_valid_url(url: str):
    """ THIS FUNCTION CHECK IS THE URL IS VALIDE """

    try:
        response = requests.head(url)

        if response.status_code == 200:
            return True
        else:
            return False

    except requests.ConnectionError:
        return False


def get_key(filename: str):

    """ 
    THIS FUNCTION TAKES IN A FILENAME, 
    PARSES AND RETURNS A KEY 
    """
    
    with open(os.path.join(os.path.join(os.getcwd(), 'query_app'), filename), 'r') as file:
        
        key = file.readline().strip()
    
    return key



def send_slack_message(webhook_url, message):
    requests.post(webhook_url, json={
        "text": message
    })


def chunk_list(codebase, num_sublists):
    chunk_size = len(codebase) // num_sublists
    codebase_chunks = [codebase[i:i + chunk_size] for i in range(0, len(codebase), chunk_size)]

    codebase_strings = []
    for chunk in codebase_chunks:
        codebase_string = '\n'.join(chunk)
        codebase_strings.append(codebase_string)

    return codebase_strings


def single_query_embedding(query: str):
    api_key = get_key(filename = "openai_key.txt")
    url = 'https://api.openai.com/v1/embeddings'
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    payload = {
                    "model": "text-embedding-ada-002",
                    "input": query
                }
    response = requests.post(url = url, headers = headers, json = payload)

    # pdb.set_trace()
    vector = json.loads(response.text)['data'][0]['embedding']
    return vector


def convert_codebase_to_embeddings(codebase: list, model: object, collection_name: str):
    
    """ 
    CONVERT CODEBASE INTO EMBEDDINGS USING 
    OPEN AI API AND THEN STORE THE EMBEDDINGS 
    IN THE DB 
    """
    num_sublists = 3000
    increment_increase = 200
    indexes = list(range(0, num_sublists, increment_increase))

    api_key = get_key(filename = "openai_key.txt")
    
    url = 'https://api.openai.com/v1/embeddings'
    
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    codebase_strings = chunk_list(codebase = codebase.split('\n'), num_sublists = num_sublists)

    for (index, num) in tqdm(enumerate(indexes)):

        try:
            start_range = indexes[index]
            end_range = indexes[index+1]
        except IndexError as error:
            break

        payload = {
                    "model": "text-embedding-ada-002",
                    "input": codebase_strings[start_range:end_range]
                    }
        response = requests.post(url = url, headers = headers, json = payload)

        for (i, text) in tqdm(enumerate(codebase_strings[start_range:end_range])):
            try:
                vector = json.loads(response.text)['data'][i]['embedding']
                embedding = model(collection_name = collection_name, string = text, embedding = str(vector))
                embedding.save()

            except IndexError as error:
                print("RAN INTO AN ERROR")
                print(text)
                


def cosine_similarity(a: list, b: list):

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)



def get_best_matches(query: str, model: object):

    """
    CONVERT INPUT STRING IN EMBEDDING USING OPEN AI API
    CHECK COSINE_SIMULARITY AGAINST DB CODEBASE EMBEDDINGS
    GET TOP 5 MATCHES STRINGS FROM DB, APPEND AND RETURN
    """
    query = single_query_embedding(query = query)
    vina_gpu = model.objects.filter(collection_name = 'vina-gpu')
    tracking = []
    # pdb.set_trace()
    for item in vina_gpu:
        # pdb.set_trace()
        embedding = vector_from_string(vector_str = item.embedding)
        score = cosine_similarity(embedding, query)
        tracking.append((score, item.string))
    
    top_5 = sorted(tracking, reverse = True)[:5]
    text_only = [x[1] for x in top_5]
    joined_text = "\n".join(text_only)
    return joined_text


def vector_from_string(vector_str: str):
    vector_str = vector_str.strip().strip('[]')
    return [float(x) for x in vector_str.split(',')]


def query_chatgpt(query: str, codebase: list):
    import pdb

    """
    THIS FUNCTION SEND A QUERY TO CHAT GPT
    """

    # API endpoint for ChatGPT
    url = "https://api.openai.com/v1/completions"

    # API key for OpenAI
    api_key = get_key(filename = "openai_key.txt")

    # Additional data/context to provide to the API
    # context = {
    #     "custom_data": {
    #         "codebase": codebase
    #     }
    # }
    # codebase_strings = chunk_list(codebase = codebase.split('\n'), num_sublists = 3000)
    # codebase = """
    #             #pragma once
    #             // Macros below are shared in both device and host
    #             #define TOLERANCE 1e-16
    #             // kernel1 macros
    #             #define MAX_NUM_OF_EVERY_M_DATA_ELEMENT 512
    #             #define MAX_M_DATA_MI 16
    #             #define MAX_M_DATA_MJ 16
    #             #define MAX_M_DATA_MK 16
    #             #define MAX_NUM_OF_TOTAL_M_DATA MAX_M_DATA_MI*MAX_M_DATA_MJ*MAX_M_DATA_MK*MAX_NUM_OF_EVERY_M_DATA_ELEMENT

    #             //kernel2 macros
    #             #define MAX_NUM_OF_LIG_TORSION 48
    #             #define MAX_NUM_OF_FLEX_TORSION 1
    #             #define MAX_NUM_OF_RIGID 48
    #             #define MAX_NUM_OF_ATOMS 130 
    #             #define SIZE_OF_MOLEC_STRUC ((3+4+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION+ 1)*sizeof(float) )
    #             """
    # pdb.set_trace()
    model = "davinci"
    # The text prompt for the API

    # for code in codebase_strings:
    # query = "What is this repo all about?"
    print(f"\nQUERY:\t{query}")

    # pdb.set_trace()

    prompt = f"Only use the below text to respond to the below query if the text is relevant to the query. Text: {codebase[:3750]} Query: {query}"

    # The request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": 'text-davinci-003',
        "prompt": f"{prompt}",
        "max_tokens": 200, # maximum number of tokens to generate in the response
        "temperature": 0.5, # control the creativity of the response
    }

    response = requests.post("https://api.openai.com/v1/completions", headers = headers, json = payload)

    if response.status_code == 200:
        # summary = response.json()["choices"][0]["text"].strip()
        # print(f"Code Summary:\n{summary}")
        print(response.json()['choices'][0]['text'])
        response = response.json()['choices'][0]['text'].split('\n\n')[-1]
        print("RESPONSE:\t", response)
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"RESPONSE:\t{response.text}\n")
    
    # pdb.set_trace()

    return response



