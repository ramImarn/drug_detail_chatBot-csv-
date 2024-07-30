import os
import re
import json
import warnings
import pandas as pd
from flask import Flask, render_template, request
from langchain.llms.openai import OpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import configration

warnings.filterwarnings('ignore')
app = Flask(__name__)
 
# Set the API keys
os.environ['OPENAI_API_KEY'] = configration.OPENAI_API_KEY 
os.environ['PINECONE_API_KEY'] = configration.PINECONE_API_KEY
 
# Initialize the Pinecone Vector Store and embeddings once to avoid repeated initialization
index_name = "drugdetails"
embeddings = OpenAIEmbeddings()
 
try:
    docSearch1 = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    print("Pinecone index loaded successfully!")
except Exception as e:
    print(f"Failed to load existing index: {e}")
 
# Initialize the OpenAI LLM and CSV agent once to avoid repeated initialization
llm = OpenAI(temperature=0)
csv_file_path = "med_train_1.csv"
agent = create_csv_agent(llm, csv_file_path, allow_dangerous_code=True)
 
@lru_cache(maxsize=100)
def spell_check(text):
    prompt = f"Correct the spelling mistakes in the following text: {text}"
    corrected_text = llm(prompt)

    return corrected_text.strip()
 
@lru_cache(maxsize=100)
def find_LLT_CODES(query):
    llt_code = []
    for k in query:
        updated_query = f"LLT_CODE for LLT_NAME {k}, only give single code as result strictly not multiple"
        response = agent.run(updated_query)
        match = re.search(r'\b\d{5,9}\b', response)
        if match:
            llt_code.append(match.group(0))
        else:
            llt_code.append(response)

    return llt_code
 
@lru_cache(maxsize=100)
def db_exts(query, j=7):
    res, sor = [], []
    rese = docSearch1.similarity_search_with_score(query, k=j)
    
    exact_match_found = False
    for i in range(j):
        res1 = re.split("LLT_NAME:", rese[i][0].page_content)[1]
        llt_name = re.sub(r"^\s+|\n:", "", res1).strip()
        
        if llt_name.lower() == query.lower() and not exact_match_found:
            res.insert(0, llt_name)
            sor.insert(0, rese[i][1])
            exact_match_found = True
        else:
            res.append(llt_name)
            sor.append(rese[i][1])
    
    if not exact_match_found:
        res = [re.sub(r"^\s+|\n:", "", re.split("LLT_NAME:", rese[i][0].page_content)[1]).strip() for i in range(j)]
        sor = [rese[i][1] for i in range(j)]

    return res, sor
 
@app.post("/")
def drug_drug():

    global word_list, score_list, word_list1
    llt_names, answer, llt_scores = [], [], []
    if request.method == 'POST':
        input_text = request.args.get('inputText')
 
        if input_text.strip().lower() == 'exit':
            return "Task complete"
        
        with ThreadPoolExecutor() as executor:
            corrected_query_future = executor.submit(spell_check, input_text)
            llt_names_future = executor.submit(db_exts, corrected_query_future.result())
            print("searching smililarity for:",corrected_query_future.result(),"wait for a moment")

        llt_names, llt_scores = llt_names_future.result()

        answer = find_LLT_CODES(tuple(llt_names))
 
        unique_results = {}
        for llt_name, llt_code, llt_score in zip(llt_names, answer, llt_scores):
            if llt_name not in unique_results:
                unique_results[llt_name] = (llt_code, llt_score)
        
        unique_llts = list(unique_results.keys())
        unique_codes = [unique_results[llt][0] for llt in unique_llts]
        unique_scores = [unique_results[llt][1] for llt in unique_llts]
 
        lst = []
        for k in range(len(unique_llts)):
            entry =  {
            "LLT_NAME": unique_llts[k],
            "LLT_CODE": unique_codes[k],
            "SIMILARITY_SCORE": unique_scores[k]
            } 
            lst.append(entry)

        json_data = json.dumps(lst)
        return json_data

if __name__ == '__main__':
    app.run(debug=True)
    
