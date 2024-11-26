import pdb

def get_host_prompts(seq, dp_text, topk, meta, args):
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        # prompt = "I've purchased the following products in the past in order: \n"
        # prompt = " "
        prompt_dp = "I've purchased the following products in the past in order: \n"
        # category_set = []
        category_set_dp = []
        for i in range(len(dp_text)):
            title_dp = dp_text[i][0]
            category_dp = dp_text[i][1]
            category_dp = category_dp.split(', ')
            category_set_dp.extend(category_dp)
            
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title_dp
            if i <= 8:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n\n'

        category_set_dp = list(set(category_set_dp))
        prompt_dp += ' The categories of these purchased products are '
        prompt_dp += ', '.join(category_set_dp)
        prompt_dp += '. \n\n'

        prompt_dp += 'Now there are 20 candidate products that I can consider to purchase next: \n'
        topk = topk.tolist()
        raw_candidate_list = []
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title
            if i <= args.topk - 1:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n'
            
            raw_candidate_list.append(title)

    elif args.dataset_code in ['ml-100k']:
        prompt_dp = "I've watched the following movies in the past in order: \n "
        category_set_dp = []
        part_history = dp_text[-10:]
        for i in range(10):
            title_dp = part_history[i][0]
            category_dp = part_history[i][1]
            category_dp = category_dp.split(', ')
            category_set_dp.extend(category_dp)
            
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title_dp
            if i <= 8:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n'

        category_set_dp = list(set(category_set_dp))
        prompt_dp += ' The genres of these watched movies are '
        prompt_dp += ', '.join(category_set_dp)
        prompt_dp += '. \n'

        prompt_dp += 'Now there are 20 candidate movies that I can watch next: \n'
        topk = topk.tolist()
        raw_candidate_list = []
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title
            if i <= args.topk - 1:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n'
            
            raw_candidate_list.append(title)

    return prompt_dp, raw_candidate_list



# def get_client_prompts(dp_text, gen_text, meta, args):
#     if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
#         prompt_dp = "The client purchased the following products in the past in order: \n"
#         category_set_dp = []

#         for i in range(len(dp_text)):
#             title_dp = dp_text[i][0]
#             category_dp = dp_text[i][1]
#             category_dp = category_dp.split(', ')
#             category_set_dp.extend(category_dp)
            
#             prompt_dp += str(i+1)
#             prompt_dp += '. '
#             prompt_dp += title_dp
#             if i <= 8:
#                 prompt_dp += ', \n'
#             else:
#                 prompt_dp += '. \n\n'

#         category_set = list(set(category_set))
#         prompt_dp += ' The categories of these purchased products are '
#         prompt_dp += ', '.join(category_set)
#         prompt_dp += '. \n'

#         prompt_dp += 'The previous agent recommended: \n'
#         prompt_dp += gen_text
    

#     elif args.dataset_code in ['ml-100k']:
#         prompt_dp = "The audience whatched the following movies in the past in order: \n "
#         category_set_dp = []

#         part_history = dp_text[-10:]
#         for i in range(10):
#             title_dp = part_history[i][0]
#             category_dp = part_history[i][1]
#             category_dp = category_dp.split(', ')
#             category_set_dp.extend(category_dp)
            
#             prompt_dp += str(i+1)
#             prompt_dp += '. '
#             prompt_dp += title
#             if i <= 8:
#                 prompt_dp += ', \n'
#             else:
#                 prompt_dp += '. \n'

#         category_set_dp = list(set(category_set_dp))
#         prompt_dp += ' The genres of these watched movies are '
#         prompt_dp += ', '.join(category_set_dp)
#         prompt_dp += '. \n'

#         prompt_dp += 'The previous agent recommended: \n'
#         prompt_dp += gen_text

#     return prompt_dp


def get_client_prompts(in_context_samples, dp_text, gen_text, meta, topk, args):
    # num_in_context_samples = len(in_context_samples)
    num_in_context_samples = 1

    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        prompt = "There is a previous client, who purchased the products you recommended: \n"

        for i in range(num_in_context_samples):
            prompt += "Client "
            prompt += str(i+1) 
            prompt += " purchased the following products in the past in order:\n"

            in_context_input = in_context_samples[i][0]
            in_context_label = in_context_samples[i][1]
        
            for j in range(len(in_context_input)):
                item = in_context_input[j]
                title = meta[item][0]
            
                prompt += str(j+1)
                prompt += '. '
                prompt += title
                if i <= 8:
                    prompt += ', \n'
                else:
                    prompt += '. \n\n'

            prompt += 'Based on these purchased products, you recommended '
            prompt += meta[in_context_label[0]][0]
            prompt += ', and Client '
            prompt += str(i+1) 
            prompt += ' purchased it. \n'

        prompt += "This means that you can understand client's preference. \
                    Now, a new client purchased the following products in the past in order: \n"
        category_set_dp = []

        for i in range(len(dp_text)):
            title_dp = dp_text[i][0]
            # category_dp = dp_text[i][1]
            # category_dp = category_dp.split(', ')
            # category_set_dp.extend(category_dp)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title_dp
            if i <= len(dp_text) - 2:
                prompt += ', \n'
            else:
                prompt += '. \n\n'

        prompt += 'There are 20 candidate products that the new client considers to purchase next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'

        # category_set = list(set(category_set))
        # prompt += ' The categories of these purchased products are '
        # prompt += ', '.join(category_set)
        # prompt += '. \n'

        if args.conversation_mode == 'polling':
            prompt += 'For this new client, a previous agent analyzed and summarized his/her preference as follows. '
            prompt += gen_text
            prompt += '. \n'

        elif args.conversation_mode == 'one_on_one':
            pass

    elif args.dataset_code in ['ml-100k']:
        prompt = "There is a previous audience, who watched the movies you recommended: \n"

        
        for i in range(num_in_context_samples):
            prompt += "Audience "
            prompt += str(i+1) 
            prompt += " watched the following movies in the past in order:\n"

            in_context_input = in_context_samples[i][0][-10:]
            in_context_label = in_context_samples[i][1]

            for j in range(len(in_context_input)):
                item = in_context_input[i]
                title_dp = meta[item][0]
                
                prompt += str(j+1)
                prompt += '. '
                prompt += title_dp
                if i <= 8:
                    prompt += ', \n'
                else:
                    prompt += '. \n\n'

            prompt += 'Based on these watched movies, you recommended '
            prompt += meta[in_context_label[0]][0]
            prompt += ', and Audience '
            prompt += str(i+1) 
            prompt += ' watched it. \n'

        prompt = "This means that you can understand audience's preference. \
                    Now, a new audience watched the following movies in the past in order: \n"
        category_set_dp = []

        for i in range(len(dp_text[-10:])):
            title_dp = dp_text[i][0]
            # category_dp = dp_text[i][1]
            # category_dp = category_dp.split(', ')
            # category_set_dp.extend(category_dp)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title_dp
            if i <= 8:
                prompt += ', \n'
            else:
                prompt += '. \n\n'

        prompt += 'There are 20 candidate movies that the new audience considers to watch next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'

        if args.conversation_mode == 'polling':
            prompt += 'For this new audience, a previous agent analyzed and summarized his/her preference as follows. '
            prompt += gen_text
            prompt += '. \n'
        
        elif args.conversation_mode == 'one_on_one':
            pass

    return prompt


def get_analyst_prompts(in_context_samples, dp_text, gen_text, meta, topk, args):
    # num_in_context_samples = len(in_context_samples)
    num_in_context_samples = 1

    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        prompt = "There is a previous client, who purchased the products you recommended: \n"

        for i in range(num_in_context_samples):
            prompt += "Client "
            prompt += str(i+1) 
            prompt += " purchased the following products in the past in order:\n"

            in_context_input = in_context_samples[i][0]
            in_context_label = in_context_samples[i][1]
        
            for j in range(len(in_context_input)):
                item = in_context_input[j]
                title = meta[item][0]
            
                prompt += str(j+1)
                prompt += '. '
                prompt += title
                if i <= 8:
                    prompt += ', \n'
                else:
                    prompt += '. \n\n'

            prompt += 'Based on these purchased products, you recommended '
            prompt += meta[in_context_label[0]][0]
            prompt += ', and Client '
            prompt += str(i+1) 
            prompt += ' purchased it. \n'

        prompt += "This means that you can understand client's preference. \
                    Now, a new client purchased the following products in the past in order: \n"
        category_set_dp = []

        for i in range(len(dp_text)):
            title_dp = dp_text[i][0]
            # category_dp = dp_text[i][1]
            # category_dp = category_dp.split(', ')
            # category_set_dp.extend(category_dp)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title_dp
            if i <= len(dp_text) - 2:
                prompt += ', \n'
            else:
                prompt += '. \n\n'

        prompt += 'There are 20 candidate products that the new client considers to purchase next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'

        if args.conversation_mode == 'polling':
            prompt += 'For this new client, can you analyze and summarize his/her preferences?'

        elif args.conversation_mode == 'one_on_one':
            pass

    elif args.dataset_code in ['ml-100k']:
        prompt = "There is a previous audience, who watched the movies you recommended: \n"

        for i in range(num_in_context_samples):
            prompt += "Audience "
            prompt += str(i+1) 
            prompt += " watched the following movies in the past in order:\n"

            in_context_input = in_context_samples[i][0][-10:]
            in_context_label = in_context_samples[i][1]
        
            for j in range(len(in_context_input)):
                item = in_context_input[i]
                title_dp = meta[item][0]
                
                prompt += str(j+1)
                prompt += '. '
                prompt += title_dp
                if i <= 8:
                    prompt += ', \n'
                else:
                    prompt += '. \n\n'

            prompt += 'Based on these watched movies, you recommended '
            prompt += meta[in_context_label[0]][0]
            prompt += ', and Audience '
            prompt += str(i+1) 
            prompt += ' watched it. \n'

        prompt = "This means that you can understand audience's preference. \
                    Now, a new audience watched the following movies in the past in order: \n"
        category_set_dp = []

        for i in range(len(dp_text[-10:])):
            title_dp = dp_text[i][0]
            # category_dp = dp_text[i][1]
            # category_dp = category_dp.split(', ')
            # category_set_dp.extend(category_dp)
            
            prompt += str(i+1)
            prompt += '. '
            prompt += title_dp
            if i <= 8:
                prompt += ', \n'
            else:
                prompt += '. \n\n'

        prompt += 'There are 20 candidate movies that the new audience considers to watch next: \n'
        topk = topk.tolist()
        for i in range(args.topk):
            item = topk[i]
            title = meta[item][0]
            prompt += str(i+1)
            prompt += '. '
            prompt += title
            if i <= args.topk - 1:
                prompt += ', \n'
            else:
                prompt += '. \n'

        if args.conversation_mode == 'polling':
            prompt += 'For this new audience, can you analyze and summarize his/her preferences? \n'
        
        elif args.conversation_mode == 'one_on_one':
            pass

    return prompt

def get_final_prompts(seq, dp_text, host_rerank_text_processed, client_rerank_text_processed_all, meta, topk, args):
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        prompt_dp = "I've purchased the following products in the past in order: \n"
        category_set_dp = []

        for i in range(len(dp_text)):
            title_dp = dp_text[i][0]
            category_dp = dp_text[i][1]
            category_dp = category_dp.split(', ')
            category_set_dp.extend(category_dp)
            
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title_dp
            if i <= 8:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n\n'

        category_set_dp = list(set(category_set_dp))
        prompt_dp += ' The categories of these purchased products are '
        prompt_dp += ', '.join(category_set_dp)
        prompt_dp += '. \n'

        prompt_dp += 'Previously, you recommended the following items for me to purchase next: \n'
        prompt_dp += host_rerank_text_processed
        prompt_dp += '. \n'

        prompt_dp += "Then, you discussed with your colleagues. They summarized my preference and made some slightly changes to the list you made before."
            
        for i in range(len(client_rerank_text_processed_all)):
            if i == 0:
                prompt_dp += "Colleage "
                prompt_dp += str(i+1) 
                prompt_dp += " analyzed my preference as follows:\n"
                prompt_dp += client_rerank_text_processed_all[i]
                prompt_dp += '\n'
            else:
                prompt_dp += "Colleage "
                prompt_dp += str(i+1) 
                prompt_dp += " recommended the following products in order:\n"
                prompt_dp += client_rerank_text_processed_all[i]
                prompt_dp += '\n'

        prompt_dp += "After the disucssion, do you want to update your recommended products? "

    elif args.dataset_code in ['ml-100k']:
        prompt_dp = "I've watched the following movies in the past in order: \n "
        category_set_dp = []

        part_history = dp_text[-10:]
        for i in range(10):
            title_dp = part_history[i][0]
            category_dp = part_history[i][1]
            category_dp = category_dp.split(', ')
            category_set_dp.extend(category_dp)
            
            prompt_dp += str(i+1)
            prompt_dp += '. '
            prompt_dp += title_dp
            if i <= 8:
                prompt_dp += ', \n'
            else:
                prompt_dp += '. \n'

        category_set_dp = list(set(category_set_dp))
        prompt_dp += ' The genres of these watched movies are '
        prompt_dp += ', '.join(category_set_dp)
        prompt_dp += '. \n'

        prompt_dp += 'Previously, you recommended the following movies for me to watch next: \n'
        prompt_dp += host_rerank_text_processed
        prompt_dp += '. \n'
    
        prompt_dp += "Then, you discussed with your colleagues. They summarized my preference and made some slightly changes to the list you made before."
            
        for i in range(len(client_rerank_text_processed_all)):
            if i == 0:
                prompt_dp += "Colleage "
                prompt_dp += str(i+1) 
                prompt_dp += " analyzed my preference as follows:\n"
                prompt_dp += client_rerank_text_processed_all[i]
                prompt_dp += '\n'
            else:
                prompt_dp += "Colleage "
                prompt_dp += str(i+1) 
                prompt_dp += " recommended the following movies in order:\n"
                prompt_dp += client_rerank_text_processed_all[i]
                prompt_dp += '\n'

        prompt_dp += "After the disucssion, do you want to update your recommended movies? "

    return prompt_dp

