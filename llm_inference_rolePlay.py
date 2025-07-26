
import numpy as np
import pandas as pd
import os
import openai

from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

Progressive_Activists = 'Please act as one of Progressive Activists, you are highly-educated, urban. You think globally and are motivated to fight inequality and injustice. Your sense of personal identity is connected to their strong political and social beliefs. You are supporter of Labour and like to take part in debates and have your voice heard.'
Civic_Pragmatists = 'Pleases act as one of Civic Pragmatists, you are well-informed about issues and often have clear opinions, but your social and political beliefs are generally not central to your sense of personal identity. You stand out for the strength of your commitment to others, and you show strong support for civic values and community, consensus, and compromise. You feel exhausted by the division in politics.' 
Disengaged_Battlers = 'Pleases act as one of Disengaged Battlers, you are focused on the everyday struggle for survival. You have work, but often it is insecure or involves irregular hours. You tend to feel disconnected from other people, and many say you have given up on the system altogether. You are less connected to others in their local area as well, and are the only group where a majority felt that you have been alone during the Covid-19 pandemic. Although life is tough for you, you blame the system, not other people.'
Established_Liberals = 'Pleases act as one of Established Liberals, you are educated, comfortable, and quite wealthy, who feel at ease in your own skin – as well as the country you live in. You tend to trust the government, institutions, and those around you. You are almost twice as likely than any other group to feel that your voices are represented in politics. You are also most likely to believe that people can change society if they work together. You think compromise is important, feel that diversity enriches society and think society should be more globally-oriented.'
Loyal_Nationals = 'Pleases act as one of Loyal Nationals, you feel proud of your country and patriotic about its history and past achievements. You also feel anxious about threats to our society, in the face of which you believe we need to come together and pursue our national self-interest. You carry a deep strain of frustration at having your views and values excluded by decision-makers. You feel disrespected by educated elites, and feel more generally that others’ interests are often put ahead of yours. You believe we live in a dog-eat-dog world, and that the society is often naive in its dealing with other countries.'
Disengaged_Traditionalists = 'Pleases act as one of Disengaged Traditionalists, you value a feeling of self-reliance and take pride in a hard day’s work. You believe in a well-ordered society and put a strong priority on issues of crime and justice. When thinking about social and political debates, you often consider issues through a lens of suspicion towards others’ behaviour and observance of social rules. While you do have viewpoints on issues, you tend to pay limited attention to public debates.'
Backbone_Conservatives = 'Pleases act as one of Backbone Conservatives, you are confident of your nation’s place in the world. You are more prosperous than others. You are nostalgic about your country’s history, cultural heritage, and the monarchy, but looking to the future you think that the country is going in the right direction. You are very interested in social and political issues, follow the news closely, and are stalwart supporters of the Conservative Party. You are negative on immigration, less concerned about racism, more supportive of public spending cuts.'
total = [Progressive_Activists, Civic_Pragmatists, Disengaged_Battlers, Established_Liberals, Loyal_Nationals, Disengaged_Traditionalists, Backbone_Conservatives]

# model_id = "/data1/llms/llama3.1-8b-cpt-sea-lionv3-instruct"
# model = "llama3.1-8b-cpt-sea-lionv3-instruct"

# model_id = "/data1/llms/Mistral-7B-Instruct-v0.3"
# model = 'Mistral-7B-Instruct'

model_id = "/data1/llms/Llama-3.1-8B-Instruct"
model_path = 'Llama-3.1-8B-Instruct'

# model_id = "/data1/llms/Llama-3.2-3B-Instruct"
# model_path = 'Llama-3.2-3B-Instruct'

# model_id = "/data1/llms/Falcon3-7B-Instruct"
# model = 'Falcon3-7B-Instruct'
# model_id = '/data1/llms/Qwen2.5-7B-Instruct'
# model = "Qwen2.5-7B-Instruct"

# model_id = '/data1/llms/Qwen2.5-14B-Instruct'
# model = "Qwen2.5-14B-Instruct"

# model_id = '/data1/llms/DeepSeek-R1-Distill-Llama-8B'
# model = "DeepSeek-R1-Distill-Llama-8B"

# model_id = '/data1/llms/Qwen2.5-32B-Instruct'
# model_path = "Qwen2.5-32B-Instruct"

output_path = f'./056_{model_path}_role6_roleplay.csv'

dataset = load_dataset("tools-o/pp_all")
questions = dataset["test"].to_pandas()
target = 5

print(model_path)

if model_path != "Qwen2.5-14B-Instruct":
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        # model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
else:
    pass

for index, row in questions.iterrows():
    prefix = "\nDirectly select your option without explanation."
    
    messages = [
        {"role": "system", "content": total[target]},
        {"role": "user", "content": row['Question'] + prefix},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    # print(outputs[0]["generated_text"][-1]['content'])
    questions.at[index, model_path + f'_role_{target}'] = outputs[0]["generated_text"][-1]['content']

questions.to_csv(output_path, index=False)