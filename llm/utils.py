def get_host_message_list(args, prompts):
    if args.dataset_code in ['beauty', 'games', 'sports', 'auto', 'toys_new']:

        system_prompt = f"""
        You are a leader of a shopping assistant team. \
        Your job is to lead your team members to make recommendations for new clients. \
        You can also make recommendations.
        """

        prompts += "Please rank these 20 products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n"
        prompts += "Please show me only your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate products. \
                    You can not generate products that are not in the given candidate list. \
                    You can not generate anything else. \n"

        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ] 

    elif args.dataset_code in ['ml-100k']:
        
        system_prompt = f"""
        You are a professional movie reviewer. \
        Your job is to lead other movie fans to recommend movies to new audiences. \
        You can also make recommendations.
        """

        prompts += "Please rank these 20 movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n"
        prompts += "Please show me your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate movies. \
                    You can not generate movies that are not in the given candidate list. \
                    You can not generate anything else. \n"

        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ]
    print("="*64)
    print("Host Start Message!!!!")
    print("="*64)
    print(prompts)
    return message_list

def get_analyst_message_list(args, prompts):
    if args.dataset_code in ['beauty', 'games', 'sports', 'auto', 'toys_new']:
        system_prompt = f"""
        You are a helpful shopping analyst. Your job is to analyze clients' perferences, \
        so that you can recommend products that match their preferences. \
        You have successfully recommended products for previous clients (Client 1). \
        You should consider your knowledge and experience with previous clients to make summarize user preferences.
        """

        prompts += "Please analyze and summarize the client's preference according to the given purchasing records.\n"
        prompts += "Please show your analysis and summary with no more than two sentences. \
                    You CAN NOT generate three or more sentences. \n"

        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ] 

    elif args.dataset_code in ['ml-100k']:
        system_prompt = f"""
        You are a helpful movie analyst. Your job is to analyze audiences' perferences, \
        so that you can recommend movies that match their preferences. \
        You have successfully recommended movies for previous audience (Client 1). \
        You should consider your knowledge and experience with previous audiences to make summarize audience preferences.
        """

        prompts += "Please analyze and summarize the audience's preference according to the given watching records.\n"
        prompts += "Please show your analysis and summary with no more than two sentences. \
                    You CAN NOT generate three or more sentences. \n"
        
        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ]
    
    print("="*64)
    print("Analyst Message!!!!")
    print("="*64)
    print(prompts)
    return message_list

def get_client_message_list(args, prompts):
    if args.dataset_code in ['beauty', 'games', 'sports', 'auto', 'toys_new']:
        system_prompt = f"""
        You are a helpful shopping assistant. Your job is to recommend products for new clients to buy. \
        You have successfully recommended products for previous clients (Client 1). \
        You should consider your knowledge and experience with previous clients to make recommendations for new clients.
        """

        prompts += "Please re-consider the ranking of these 20 products by measuring the possibilities that the new client would like to purchase next most, according to the given purchasing records. \
                    Please consider your recommendation based on the analysis and summary from the previous agent.\
                    Think step by step. \n"
        prompts += "Please show your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate products. \
                    You can not generate products that are not in the given candidate list. \
                    You can not generate anything else. \n"

        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ] 

    elif args.dataset_code in ['ml-100k']:
        system_prompt = f"""
        You are a movie fan. Your job is to recommend movies for new audiences. \
        You have successfully recommended movies for previous audience (Audience 1). \
        You should consider your knowledge and experience with previous audience to make recommendations for new audiences.
        """

        prompts += "Please re-consider the ranking of these 20 movies by measuring the possibilities that the new audience would like to watch next most, according to the given watching records. \
                    Please consider your recommendation based on the analysis and summary from the previous agent.\
                    Think step by step. \n"

        prompts += "Please show your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate movies. \
                    You can not generate movies that are not in the given candidate list. \
                    You can not generate anything else. \n"
                    
        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ]
    
    print("="*64)
    print("Client Message!!!!")
    print("="*64)
    print(prompts)
    return message_list

def get_final_message_list(args, prompts):
    if args.dataset_code in ['beauty', 'games', 'sports', 'auto', 'toys_new']:
        system_prompt = f"""
        You are a leader of a shopping assistant team. \
        Your job is to lead your team members to make recommendations for new clients. \
        You can also make recommendations. \
        You discussed with your shopping assistant colleages, and now it is time for you to make the final decision. \
        You should consider your original recommendations, and the recommendations from your colleages to make recommendations for the new client.
        """

        prompts += "Please re-consider the ranking of the 20 recommended products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. \
                    Please consider your recommendation based on your original recommendation, the user preference analysis and the recommenations from your colleages. Think step by step.\n"
        prompts += "Please show your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate products. You can not generate products that are not in the given candidate list.\n"

        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ] 

    elif args.dataset_code in ['ml-100k']:
        system_prompt = f"""
        You are a professional movie reviewer. \
        Your job is to lead other movie fans to recommend movies to new audiences. \
        You can also make recommendations. \
        You discussed with some other movie fans, and now it is time for you to make the final decision. \
        You should consider your original recommendations, and the recommendations from other movie fans to make recommendations for the new audience.
        """

        prompts += "Please re-consider the ranking of the 20 recommended movies by measuring the possibilities that I would like to watch next most, according to the given watch records. \
                    Please consider your recommendation based on your original recommendation, the user preference analysis and the recommenations from other movie fans. Think step by step.\n"
        prompts += "Please show your ranking results with order numbers. Split your output with line break. \
                    You MUST rank the given candidate products. You can not generate products that are not in the given candidate list.\n"
        
        message_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts},
                    ]
    
    print("="*64)
    print("Final Host Message!!!!")
    print("="*64)
    print(prompts)

    return message_list