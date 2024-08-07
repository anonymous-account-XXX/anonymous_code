import json
import pandas as pd
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

import openai
from openai import OpenAI

def callChatGPT(prompt):
    client = OpenAI(api_key=API_KEY)
    system = "You are a data augmentation model that response following the exactly same format as the provide example."
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[
                                                  {"role": "system", "content": system},
                                                  {"role": "user", "content": prompt}],
                                              temperature=0.7,
                                              max_tokens=1000)

    chatgpt_pred = response.choices[0].message.content
    return chatgpt_pred
  
  
def social_support_assessment(data):
    with open('./social_support.txt') as f:
        instructions = f.readlines()
        
    prompt_ = ''.join(instructions)
    prompt_ = prompt_.replace('<<AGE>>', str(data['age'])) 
    prompt_ = prompt_.replace('<<MaritalStatus>>', data['marital_status'])
    prompt_ = prompt_.replace('<<Number>>', str(data['number_of_records']))
    prompt_ = prompt_.replace('<<ArrivalType>>', data['admit_type'])
    prompt_ = prompt_.replace('<<ArrivalWay>>', data['admit_location'])
    
    response = callChatGPT(prompt_)
    return response
  
def social_support_assessment_train(data):
    with open('./social_support_training.txt') as f:
        instructions = f.readlines()
        
    prompt_ = ''.join(instructions)
    prompt_ = prompt_.replace('<<DISCHARG>>', str(data['discharge_location']))
    prompt_ = prompt_.replace('<<AGE>>', str(data['age'])) 
    prompt_ = prompt_.replace('<<MaritalStatus>>', data['marital_status'])
    prompt_ = prompt_.replace('<<Number>>', str(data['number_of_records']))
    prompt_ = prompt_.replace('<<ArrivalType>>', data['admit_type'])
    prompt_ = prompt_.replace('<<ArrivalWay>>', data['admit_location'])
    
    response = callChatGPT(prompt_)
    return response

def gpt_finetune_schema_reformat(prompt, response):
    system = "You are a healthcare professional assessing a patient's social support to understand their overall well-being and potential factors influencing their final discharge plan."
    schema = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}
    return schema
    
def generate_finetune_data():
    data = pd.read_csv('../data/train_subset_data.csv')
    os.makedirs("../data/rationales", exist_ok=True)
    
    rationales = []
    for i in tqdm(range(len(data[:10]))):
        prompt = social_support_assessment_train(data.iloc[i])
        response = callChatGPT(prompt)
        
        # quality check
        flag = 0
        res = response.split('Result (Only choose from Weak or Strong. Force to choose one):\n')[1].strip().lower()
        if data.iloc[i]['discharge_location'] == "home":
            if res == 'strong':
                flag = 1
            else:
                flag = 0
        else:
            if res == 'weak':
                flag = 1
            else:
                flag = 0
                
        if flag == 1:
            real_prompt = social_support_assessment(data.iloc[i])
            rationales.append(gpt_finetune_schema_reformat(real_prompt, response))
    
    with open('../data/rationales/social_support_train.jsonl', 'w') as f:
        for rationale in rationales:
            f.write(json.dumps(rationale))
            f.write('\n')
            
def finetune():
    client = OpenAI(api_key=API_KEY)
    client.fine_tuning.jobs.create(
        training_file='../data/rationales/social_support_train.jsonl', 
        model="gpt-4o-mini", 
        hyperparameters={
            "n_epochs":3
        })
    
if __name__ == '__main__':
    # prepare the training data
    generate_finetune_data()
    # finetune the model
    finetune()
        

