import json
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

import openai
from openai import OpenAI

client = OpenAI(api_key=API_KEY)

#TODO - please use the finetuned model name
finetuned_model = "ft:gpt-4o-mini:my-org:custom_suffix:id"

def callChatGPT(prompt):
    system_prompt = "You are a healthcare professional assessing a patient's social support to understand their overall well-being and potential factors influencing their final discharge plan. "
    response = client.chat.completions.create(model=finetuned_model,
                                              messages=[
                                                  {"role": "system", "content": system_prompt},
                                                  {"role": "user", "content": prompt}],
                                              temperature=0.7,
                                              max_tokens=1000)
    chatgpt_pred = response.choices[0].message.content
    return chatgpt_pred
  
  
def social_support_assessment(data):
    with open('./prompt/social_support.txt') as f:
        instructions = f.readlines()
        
    prompt_ = ''.join(instructions)
    prompt_ = prompt_.replace('<<AGE>>', str(data['age'])) 
    prompt_ = prompt_.replace('<<MaritalStatus>>', data['marital_status'])
    prompt_ = prompt_.replace('<<Number>>', str(data['number_of_records']))
    prompt_ = prompt_.replace('<<ArrivalType>>', data['admit_type'])
    prompt_ = prompt_.replace('<<ArrivalWay>>', data['admit_location'])
    
    response = callChatGPT(prompt_)
    return response
    

def process_patient(args):
    index, data = args
    patient_id = str(data['id'][index])
    response = social_support_assessment(data.iloc[index])
    result = {patient_id: response}
    return result


def parallel_social_support_assessment(data, max_workers=6):
    with Pool(max_workers) as pool:
        results = []
        with tqdm(total=len(data)) as pbar:
            for result in pool.imap_unordered(process_patient, [(i, data) for i in range(len(data))]):
                results.append(result)
                pbar.update()
                with open ('./results/temp_results.json', 'w') as f:
                    json.dump(results, f)
    return results
  
    
def pipeline():
  data = pd.read_csv('../data/test_subset_data.csv')
  os.makedirs("./results", exist_ok=True)
  
  #infer social support
  results = parallel_social_support_assessment(data)
      
  #post-process the results  
  all_results = {}
  for result in results:
      key = list(result.keys())[0]
      value = result[key]
      all_results[key] = value
       
  social_support = []
  for i in tqdm(range(data.shape[0])):
    id = str(data['id'][i])
    response = all_results[id]
    res = response.split("Result (Only choose from Weak or Strong. Force to choose one):\n")[1].strip().lower()
    social_support.append("Strong" if "strong" in res else "Weak")
    
  data['social_support'] = social_support
  unique_values = data['social_support'].unique()
  print(unique_values)
  data.to_csv('./results/subset_with_social_support.csv', index=False)
      
    

if __name__ == '__main__':
    pipeline()
  
    
    
  



        

