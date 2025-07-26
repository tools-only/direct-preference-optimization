
import numpy as np
import pandas as pd
import os
import openai

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
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

# model_id = "/data1/llms/Llama-3.2-3B-Instruct"
# model_path = 'Llama-3.2-3B-Instruct-wdpo'

# model_id = "/data1/llms/Llama-3.2-3B-Instruct"
# policy_id = "/data1/llms/DPO/Llama-3.2-3B-Instruct_DPO/LATEST/policy.pt"
# policy_id = "/data1/llms/cache/Llama-3.2-3B-Instruct_2025-04-15_20-18-41_813226/LATEST/policy.pt"
# policy_id = "/data1/llms/WDPO-small-reason/Llama-3.2-3B-Instruct_2025-04-16_14-14-09_120542/LATEST/policy.pt"
# model_path = 'Llama-3.2-3B-Instruct-wdpo-small-reason'

# policy_id = "/data1/llms/WDPO-small/Llama-3.2-3B-Instruct_2025-04-16_14-27-57_760315/LATEST/policy.pt"
# policy_id = "/data1/llms/WDPO-small-reason/Llama-3.2-3B-Instruct_2025-04-16_21-44-00_798623/LATEST/policy.pt"
# model_path = 'Llama-3.2-3B-Instruct-wdpo-small'

# model_id = "/data1/llms/Falcon3-7B-Instruct"
# model = 'Falcon3-7B-Instruct'
# model_id = '/data1/llms/Qwen2.5-7B-Instruct'
# model_path = "Qwen2.5-7B-Instruct"

# model_id = '/data1/llms/Qwen2.5-14B-Instruct'
# model = "Qwen2.5-14B-Instruct"

# model_id = '/data1/llms/DeepSeek-R1-Distill-Llama-8B'
# model = "DeepSeek-R1-Distill-Llama-8B"

# model_id = '/data1/llms/Qwen2.5-32B-Instruct'
# model_path = "Qwen2.5-32B-Instruct"

# file_path = './formal_5_fair_necessities_choices.csv'
model_id = "/data1/llms/Llama-3.2-3B-Instruct"
# running_list = ['normal', 'dpo-small', 'dpo-small-reason', 'dpo-small-reason-constrative', 'WDPO-small', 'WDPO-small-reason', 'WDPO-small-reason-constrative']
# running_list = ['WDPO-small', 'WDPO-small-reason', 'WDPO-small-reason-constrative']
# running_list = ['dpo-all']
# running_list = ['wdpo-all-reason']
# running_list=['wdpo-small-reason-summary']
# running_list = ['normal', 'dpo-all', 'WDPO', 'wdpo-all-reason', 'wdpo-small-reason-summary']

# data="news"
# data = "test"
data = "simulation"
# running_list = ['normal', 'dpo-all', 'WDPO', 'wdpo-all-reason', 'wdpo-small-reason-summary', 'dpo-all-role6', 'wdpo-all-role6']
# loss = 'sft'
loss = 'dpo'
# loss = 'normal'
# loss = 'role-play'
role = 6 # 6
# role = 'all'
# running_list = ['dpo', 'wdpo']
running_list = ['normal', 'role-play', 'dpo', 'wdpo']
# running_list = ['wdpo', 'role-play']

print(data, loss, role, running_list)
# running_list = ['wdpo-all-reason', 'wdpo-small-reason-summary', 'dpo-all-role6', 'wdpo-all-role6']
# running_list = ['WDPO']
# running_list = ['dpo', 'wdpo', 'dpo-all-reason', 'dpo-reason-small', 'wdpo_reason']
# running_list = ['dpo', 'wdpo', 'wdpo_reason']
# running_list = ['WDPO', 'WDPO-reason']

for m in running_list:
    if loss == 'sft':
        if role == 2:
            policy_id = f"/data1/llms/sft/{m}/policy.pt"
        if role == 6:
            policy_id = f"/data1/llms/sft-6/{m}/policy.pt"
        if role == 5:
            policy_id = f"/data1/llms/sft-5/{m}/policy.pt"
    elif loss == 'dpo':
        if role == 2:
            policy_id = f"/data1/llms/dpo-role{role}/{m}/policy.pt"
        if role == 6:
            policy_id = f"/data1/llms/dpo-role{role}/{m}/policy.pt"
        if role == 5:
            policy_id = f"/data1/llms/dpo-role{role}/{m}/policy.pt"
    model_path = f'Llama-3.2-3B-Instruct-{m}'
    output_path = f'./516_{loss}_hf_{data}_{model_path}_{role}.csv' # simulation

    print(model_path)
    print(output_path)

    if model_path != "Qwen2.5-14B-Instruct" and 'dpo' not in model_path.lower():
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
    elif 'dpo' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        model.load_state_dict(torch.load(policy_id)['state'])
    else:
        pass

    if 'dpo' in model_path.lower():
        # load test data from huggingface
        if data == 'test':
            dataset = load_dataset("tools-o/pp_all", cache_dir='/home/zq/nips2025/')
            questions = dataset["test"].to_pandas()
            prefix = "\nDirectly select your option without explanation."
            name = 'Question'
        elif data == 'simulation':
            dataset = load_dataset("tools-o/simulation_test", cache_dir=None)
            questions = dataset["train"].to_pandas()
            prefix = ""
            name = 'simulation_question'
        elif data == 'news':
            # dataset = load_dataset("tools-o/news_insights")
            # questions = dataset["train"].to_pandas()
            dataset = pd.read_csv('news.csv')
            questions = dataset
            prefix = "**Instructions**:\n- From your role's perspective, provide a concise summary (2-3 sentences maximum)\n- Highlight 2-3 key points that would be most relevant to your role\n- Keep the response focused and role-specific\n- Use clear bullet points for structure\n**Expected Output Format**:\n[Summary]: (Brief overview)\n[Key Points]:\n1. (First role-relevant point)\n2. (Second role-relevant point)\n3. (Optional third point)\n[Role-Specific Insight]: (1 sentence perspective)"
            name = 'news_truncated'

        for index, row in questions.iterrows():
            if data == 'news':
                messages = [
                    {"role": "user", "content": "Given the following article:\n" + row[name] + prefix},
                ]
            else:
                messages = [
                    {"role": "user", "content": row[name] + prefix},
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            questions.at[index, model_path] = response

        questions.to_csv(output_path, index=False)

    else:
        if data == 'test':
            dataset = load_dataset("tools-o/pp_all", cache_dir=None)
            questions = dataset["test"].to_pandas()
            prefix = "\nDirectly select your option without explanation."
            name = 'Question'
        elif data == 'simulation':
            dataset = load_dataset("tools-o/simulation_test", cache_dir=None)
            questions = dataset["train"].to_pandas()
            name = 'simulation_question'
            prefix = ""
        elif data == 'news':
            dataset = load_dataset("tools-o/news_insights")
            questions = dataset["train"].to_pandas()
            prefix = "**Instructions**:\n- From your role's perspective, provide a concise summary (2-3 sentences maximum)\n- Highlight 2-3 key points that would be most relevant to your role\n- Keep the response focused and role-specific\n- Use clear bullet points for structure\n**Expected Output Format**:\n[Summary]: (Brief overview)\n[Key Points]:\n1. (First role-relevant point)\n2. (Second role-relevant point)\n3. (Optional third point)\n[Role-Specific Insight]: (1 sentence perspective)"
            name = 'news_truncated'

        for index, row in questions.iterrows():
            if data == 'news':
                if m != 'role-play':
                    messages = [
                        {"role": "user", "content": "Given the following article:\n" + row[name] + prefix},
                    ]
                elif m == 'role-play':
                    messages = [
                        {"role": "system", "content": total[role - 1]},
                        {"role": "user", "content": "Given the following article:\n" + row[name] + prefix}
                    ]
            else:
                if m != 'role-play':
                    messages = [
                        {"role": "user", "content": row[name] + prefix},
                    ]
                elif m == 'role-play':
                    messages = [
                        {"role": "system", "content": total[role - 1]},
                        {"role": "user", "content": row[name] + prefix},
                    ]

            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            questions.at[index, model_path] = outputs[0]["generated_text"][-1]['content']

        questions.to_csv(output_path, index=False)


        # if model_path == "Qwen2.5-32B-Instruct":
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     torch_dtype="auto",
            #     device_map="auto"
            # )
            # tokenizer = AutoTokenizer.from_pretrained(model_id)

            # questions = pd.read_csv(file_path)
            # for index, question in enumerate(questions['Question']):
            #     for role_index, role in enumerate(total):
            #         value = questions.at[index, f'gpt-4o-mini_role_{role_index + 1}']
            #         reason = f"Given the question: {question}\nConcisely summarize the key argument that best supports the answer: {value}. Please ensure that your answer is consistent with your role."

            #         messages = [
            #             {"role": "system", "content": role},
            #             {"role": "user", "content": reason}
            #         ]
            #         text = tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True
            #         )
            #         model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            #         generated_ids = model.generate(
            #             **model_inputs,
            #             max_new_tokens=512
            #         )
            #         generated_ids = [
            #             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            #         ]

            #     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #     questions.at[index, f'{model}_role_{role_index + 1}_reason'] = response
            #     if index % 100 == 0:
            #         questions.to_csv('formal_5_fair_necessities_roles_reason.csv', index=False) 
            # questions.to_csv('formal_5_fair_necessities_roles_reason.csv', index=False)
