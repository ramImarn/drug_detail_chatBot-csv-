import os
import pandas as pd
from langchain.llms.openai import OpenAI 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.runnables.base import RunnableSequence, Runnable
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time

import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import re

import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def drug_drug():
    global word_list,score_list,word_list1

    csv_file_path = "med_train_1.csv"
    #csv_file_path = "/home/raavan/Downloads/meddra261_full_data.csv"
    os.environ['OPENAI_API_KEY'] = "enter your openai api key"

    os.environ['PINECONE_API_KEY'] = "enter your pinecone api key"
    index_name = "drugdetails"
    embeddings = OpenAIEmbeddings()

    try:
        # Load existing index
        docSearch1 = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        print("searching similar results wait for a moment!!!")

    except Exception as e:
        print(f"Failed to load existing index: {e}")



    def find_LLT_CODES (query):
        llt_code=[]
        llm = OpenAI(temperature=0)
        agent = create_csv_agent(llm, csv_file_path, allow_dangerous_code=True)
            
        for k in query:
            updated_query = f"LLT_CODE for LLT_NAME {k},only give single code as result strectly not multiple"
            llt_code.append(agent.run(updated_query))
        return llt_code


    def db_exts (query,j=9):
        res,sor=[],[]
        rese = docSearch1.similarity_search_with_score(query,k=j)
        for i in range(j):
            res.append(re.split("LLT_NAME:",re.sub("\n:","",rese[i][0].page_content))[1])
            sor.append(rese[i][1])
        return res,sor

    words_list = []
    scores_list = []
    words_list1 = []
    
    llt_names,answer,llt_scores=[],[],[]
    if request.method == 'POST':
        input_text = request.form['inputText']

        if input_text.strip().lower() == 'exit':
            return render_template('table.html', word_list = words_list, score_list = scores_list, word_list1 = words_list1)
        
        query = input_text
        
        llt_names,llt_scores = db_exts (query)
        answer = find_LLT_CODES (llt_names)

        words_list = llt_names
        scores_list = answer
        words_list1 = llt_scores
    
    max_length = max(len(words_list), len(scores_list), len(words_list1))  # Calculate the maximum length of the lists

    return render_template('table.html', word_list = words_list, score_list = scores_list, word_list1 = words_list1, max_length = max_length)
 
 
if __name__ == '__main__':
    app.run(debug=True)