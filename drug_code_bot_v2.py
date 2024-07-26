import os
import re
import warnings
import configration
import pandas as pd
from spellchecker import SpellChecker
from langchain.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.agents import create_csv_agent
 
warnings.filterwarnings('ignore')
 
app = Flask(__name__)
 
@app.route("/", methods=['GET', 'POST'])
def drug_drug():
    global word_list, score_list, word_list1
 
    #csv_file_path = "med_train_1.csv"
    csv_file_path = configration.CSV_FILE_PATH
    os.environ['OPENAI_API_KEY'] = configration.OPENAI_API_KEY  
    os.environ['PINECONE_API_KEY'] = configration.PINECONE_API_KEY     
    index_name = configration.INDEX_NAME
    embeddings = OpenAIEmbeddings()
 
    try:
        # Load existing index
        docSearch1 = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        print("Searching similar results, wait for a moment!!!")
    except Exception as e:
        print(f"Failed to load existing index: {e}")
 
    def find_LLT_CODES(query):
        llt_code = []
        llm = OpenAI(temperature=0)
        agent = create_csv_agent(llm, csv_file_path, allow_dangerous_code=True)
        
        for k in query:
            updated_query = f"LLT_CODE for LLT_NAME {k}, only give single code as result strictly not multiple"
            response = agent.run(updated_query)
            match = re.search(r'\b\d{5,9}\b', response)
            if match:
                llt_code.append(match.group(0))
            else:
                llt_code.append(response)
        return llt_code
 
    def db_exts(query, j=15):
        spell = SpellChecker()
        try:
            if query:
                corrected_query = spell.correction(query.strip())
            else:
                corrected_query = ''
        except Exception as e:
            print(f"Spell correction error: {e}")
            corrected_query = query

        res, sor = [], []
        rese = docSearch1.similarity_search_with_score(query, k=j)
        
        exact_match_found = False
        for i in range(j):
            res1 = re.split("LLT_NAME:", rese[i][0].page_content)[1]
            llt_name = re.sub(r"^\s+|\n:", "", res1).strip()
            
            if llt_name.lower() == query.lower() and not exact_match_found:
                # Insert exact match at the beginning
                res.insert(0, llt_name)
                sor.insert(0, rese[i][1])
                exact_match_found = True
            else:
                res.append(llt_name)
                sor.append(rese[i][1])
        
        if not exact_match_found:
            # If no exact match found, keep the original order
            res = [re.sub(r"^\s+|\n:", "", re.split("LLT_NAME:", rese[i][0].page_content)[1]).strip() for i in range(j)]
            sor = [rese[i][1] for i in range(j)]
        
        return res, sor
 
    words_list = []
    scores_list = []
    words_list1 = []
    
    llt_names, answer, llt_scores = [], [], []
    if request.method == 'POST':
        input_text = request.form['inputText']
 
        if input_text.strip().lower() == 'exit':
            return render_template('table.html', word_list=words_list, score_list=scores_list, word_list1=words_list1)
        
        query = input_text
        
        llt_names, llt_scores = db_exts(query)
        answer = find_LLT_CODES(llt_names)
 
        unique_results = {}
        for llt_name, llt_code, llt_score in zip(llt_names, answer, llt_scores):
            if llt_name not in unique_results:
                unique_results[llt_name] = (llt_code, llt_score)
        
        unique_llts = list(unique_results.keys())
        unique_codes = [unique_results[llt][0] for llt in unique_llts]
        unique_scores = [unique_results[llt][1] for llt in unique_llts]
 
        words_list = unique_llts[:5]
        scores_list = unique_codes[:5]
        words_list1 = unique_scores[:5]
    
    max_length = max(len(words_list), len(scores_list), len(words_list1))
 
    return render_template('table.html', word_list=words_list, score_list=scores_list, word_list1=words_list1, max_length=max_length)
 
if __name__ == '__main__':
    app.run(debug=True)