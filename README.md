# drug_detail_chatBot-csv
langchain_pinecone model

-> This project use langchain framework with 2 agents.

-> This agent use csv file as source dataset and process it with OpenAI to convert it as embeddings.

-> Then store the embeddings with pinecone database.

 **Agent I**
 
   -> Takes the user query (i have headache) as input to this agent.
   
   -> This agent proess the input with langchain and checks for the similarities from the database.
   
   -> It return the result to Agent II, as top 9 matching drugs which all related to user query (headache).

 **Agent II**
 
   -> Takes Agent I result and process it for finding respective code for the drug.
   
   -> Checks the dataset from csv and return the code with similarity score similar drug and its code.
   
   -> send it to thr front end UI through the flask API.
    
**workflow**
![Drugbot_flowchart](https://github.com/user-attachments/assets/8477a104-8e36-4d9e-8c71-f9113899a7f5)

**sample output**
![Screenshot from 2024-07-11 19-14-25](https://github.com/user-attachments/assets/47d31817-723c-40be-93e1-7ff8e7f7e0f3)

**Installation**

-> create a new environment (python3.10).

-> clone the repo.

-> install requirements (pip install requirements.txt)

-> Edit drug_code_bot.py file and enter your openai and pinecone key and save it.

-> run main file (flask --app drug_code_bot.py run)
