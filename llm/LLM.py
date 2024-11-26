from autogen import oai
import openai

from .LLM import *
from .utils import *
from .prompts import *

import numpy as np

def conversation(seq, seq_dp, dp_text, topk, topk_dp, meta, in_context_samples, args):
    openai.api_key = args.openai_token
    # model = 'gpt-4o-2024-05-13'
    model = 'gpt-3.5-turbo-0125'

    conversation_text = {}
    # get host prompts
    # conversation starts with the central agent
    prompts, raw_candidate_list = get_host_prompts(seq, dp_text, topk, meta, args)
    message_list = get_host_message_list(args, prompts)

    # create a chat completion request
    response = oai.ChatCompletion.create(
        # config_list=config_list_gpt4,
        model=model,
        messages=message_list,
        # request_timeout=300,  # may be necessary for larger models
    )
    print("="*64)
    print('host Response {}:\n'.format(1))
    print("="*64)
    print(response.choices[0].message.content, '\n\n')
    host_rerank_text = response.choices[0].message.content
    host_rerank_text_processed = process_rerank_text(host_rerank_text)
    conversation_text["host_start"] = host_rerank_text_processed
    
    client_rerank_text_all, client_rerank_text_processed_all = local_conversation(model, dp_text, host_rerank_text_processed, in_context_samples, meta, topk, args)
    conversation_text["client_dicussion"] = client_rerank_text_processed_all

    final_rerank_text, rerank_final_processed = final_conversation(model, seq, dp_text, host_rerank_text_processed, client_rerank_text_processed_all, meta, topk, args)
    conversation_text["final_discussion"] = rerank_final_processed

    # pdb.set_trace()
    return conversation_text

def local_conversation(model, dp_text, rerank_text_processed, in_context_samples, meta, topk, args):
    rerank_text_all = []
    rerank_text_processed_all = []
    
    selected_client_ids = np.random.choice(args.num_clients, args.num_clients // 2, replace=False).tolist()
    # for idx in range(args.num_clients):
    counter = 0
    for idx in sorted(selected_client_ids):
        if counter == 0:
            client_prompts = get_analyst_prompts(in_context_samples[idx], dp_text, rerank_text_processed, meta, topk, args)
            client_message_list = get_analyst_message_list(args, client_prompts)

        else:
            # in_context_sample = in_context_all[np.random.choice(len(in_context_feature_ids), 1, replace=False)]
            client_prompts = get_client_prompts(in_context_samples[idx], dp_text, rerank_text_processed, meta, topk, args)
            client_message_list = get_client_message_list(args, client_prompts)
        
        # create a chat completion request
        response = oai.ChatCompletion.create(
            # config_list=config_list_gpt4,
            model=model,
            messages=client_message_list,
            temperature = 0
            # request_timeout=300,  # may be necessary for larger models
        )
        print("="*64)
        print('Client Response {}:\n'.format(idx+1))
        print("="*64)
        print(response.choices[0].message.content, '\n\n')
        rerank_text = response.choices[0].message.content
        
        if counter == 0:
            rerank_text_processed = rerank_text
            rerank_text_all.append(rerank_text)
            rerank_text_processed_all.append(rerank_text)
        else:
            rerank_text_processed = process_rerank_text(rerank_text)
            rerank_text_all.append(rerank_text)
            rerank_text_processed_all.append(rerank_text_processed)

        # pdb.set_trace()
    return rerank_text_all, rerank_text_processed_all

def final_conversation(model, seq, dp_text, host_rerank_text_processed, client_rerank_text_processed_all, meta, topk, args):
    prompts = get_final_prompts(seq, dp_text, host_rerank_text_processed, client_rerank_text_processed_all, meta, topk, args)
    message_list = get_final_message_list(args, prompts)
    
    # create a chat completion request
    response = oai.ChatCompletion.create(
        # config_list=config_list_gpt4,
        model=model,
        messages=message_list,
        # request_timeout=300,  # may be necessary for larger models
    )
    print("="*64)
    print('Final Host Response {}:\n'.format(1))
    print("="*64)
    print(response.choices[0].message.content, '\n\n')
    rerank_text = response.choices[0].message.content
    rerank_text_processed = process_rerank_text(rerank_text)

    return rerank_text, rerank_text_processed

def process_rerank_text(rerank_text):
    re_rank_list = rerank_text.split('\n')
    rerank_text_processed = r""
    raw_rerank_list = []
    counter = 1
    for item_text in re_rank_list:
        # if '. ' in item_text:
        #     old_id_text = item_text.split(". ")[-1]
        #     old_id = int(old_id_text) - 1
        #     if old_id > 19:
        #         old_id = np.random.choice(20,1)[0]
        #     raw_rerank_list.append(int(old_id_text))
            
        # else:
        #     old_id = int(item_text) - 1
        #     if old_id > 19:
        #         old_id = np.random.choice(20,1)[0]
        #         raw_rerank_list.append(old_id)
        #     else:
        #         raw_rerank_list.append(int(item_text))

        if '. ' in item_text:
            rerank_text_processed += item_text
            rerank_text_processed += '\n'
            
    return rerank_text_processed