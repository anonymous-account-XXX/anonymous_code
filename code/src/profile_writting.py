import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

import json
import openai
from openai import OpenAI

client = OpenAI(api_key=API_KEY)

def callChatGPT(prompt):
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[
                                                  {"role": "user", "content": prompt}],
                                              temperature=0.7,
                                              max_tokens=1000)

    chatgpt_pred = response.choices[0].message.content
    return chatgpt_pred


def generate_profile(patient_data):
    prompt = (
        #TODO: Add your prompt here (Due to privacy concerns, we cannot provide the prompt)
    )
    for record in patient_data['records']:
        prompt += (
            # TODO: Add your prompt here (Due to privacy concerns, we cannot provide the prompt)
        )
    
    response = callChatGPT(prompt)
    
    return response

with open('patient_records.json') as f:
    patient_data_list = json.load(f)

profiles = {}
for patient_data in patient_data_list:
    profile = generate_profile(patient_data)
    profiles[patient_data['id']] = profile
    
with open('patient_profiles.json', 'w') as f:
    json.dump(profiles, f)
    
