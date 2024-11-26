import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import *
from model import *
from dataloader import *
from trainer import *
import copy 

from pytorch_lightning import seed_everything

from TTA import *

        
def retrieve_in_context_examples(model_e5, model_lru, client_data, test_data, dp_text, meta, save_root, args):
    e5_val_user_embeddings = generate_e5_user_embeddings(client_data, model_e5, meta, args)
    lru_val_user_embeddings = generate_lru_user_embeddings(client_data, model_lru, args)
    
    # prepare test dataloader
    test_dataset = E5DP_textTestDataset(args, test_data, args.bert_max_len, dp_text, meta)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
    #                                             shuffle=False, pin_memory=True, num_workers=args.num_workers,
    #                                             collate_fn=collate_fn_dp)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, pin_memory=True, num_workers=0,
                                                collate_fn=collate_fn_dp)
    users = sorted(client_data[0].keys())
    users = np.array([u for u in users if len(client_data[1][u]) > 0])

    in_context_all = [] # in-context all
    for batch_idx, batch in enumerate(test_loader):
        in_context_feature_ids = hybrid_retrival(model_e5, model_lru, batch, meta, e5_val_user_embeddings, lru_val_user_embeddings, args)
        batch_size = len(in_context_feature_ids)
        for i in range(batch_size):
            cur_feature_id = in_context_feature_ids[i]
            in_context_user_ids = users[cur_feature_id]
            in_context_sample_per_user = []
            for in_context_user_id in in_context_user_ids:
                ic_input = client_data[0][in_context_user_id]
                ic_label = client_data[1][in_context_user_id]
                in_context_sample_per_user.append([ic_input, ic_label])
            in_context_all.append(in_context_sample_per_user)

    with open(os.path.join(save_root, "in_context_samples.pkl"), 'wb') as f:
        pickle.dump({'in_context_samples': in_context_all}, f)

def generate_lru_user_embeddings(client_data, model, args):
    train_data = client_data[0]    
    val_data = client_data[1]
    val_dataset = LRUValidDataset(args, train_data, val_data, args.bert_max_len)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)
    user_features_his = []
    test_scores = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        print('****************** Generating LRU user embeddings ******************')
        tqdm_dataloader = tqdm(val_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.cuda() for x in batch]
            seqs, labels = batch
            batch_size = len(seqs)
            model_features = model.forward_feature(seqs)
            
            # scores = torch.matmul(model_features, model.embedding.token.weight.permute(1, 0)) + model.model.bias
            # scores = scores[:, -1, :]
            # scores[:, 0] = -1e9  # padding
            # rank = (-scores).argsort(dim=1)  
            # rank = rank[:, :args.topk_user]
            # probs = []
            # for i in range(batch_size):
            #     scores_topk = scores[i][rank[i]]
            #     prob = torch.softmax(scores_topk.unsqueeze(0), dim=-1)
            #     probs.append(prob)
            # probs = torch.cat(probs, dim=0)
            # entropy = -torch.sum(probs * torch.log(probs), dim=-1)

            # # filter out uncertain users
            # p = 1 / args.topk_user
            # p_random_guess = torch.tensor([p for _ in range(args.topk_user)])
            # max_entropy = -torch.sum(p_random_guess * torch.log(p_random_guess))
            # xi = max_entropy.item() * args.uncertainty
            # selected_ids = torch.where((entropy < xi) != 0)[0]

            user_features = model_features[:, -1, :]
            # [selected_ids]
            user_features_his.append(user_features)

    user_features_his = torch.cat(user_features_his,dim=0)

    return user_features_his

def generate_e5_user_embeddings(client_data, model, meta, args):
    train_data = client_data[0]
    val_data = client_data[1]
    val_dataset = E5ValidDataset(args, train_data, val_data, args.bert_max_len, meta)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                                    shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                                    collate_fn=collate_fn)
    user_features_his = []   
    model.eval()     
    with torch.no_grad():
        candidate_embeddings = calculate_all_item_embeddings(model, meta, args)

        print('****************** Generating E5 user embeddings ******************')
        tqdm_dataloader = tqdm(val_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            input_prompts = get_batch_prompts(args, batch, meta)
            batch_size = len(batch[0])

            input_tokens = model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
            input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

            # forward pass
            outputs = model.model(**input_tokens)
            embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
            embeddings = F.normalize(embeddings, dim=-1)

            user_features_his.append(embeddings)

    user_features_his = torch.cat(user_features_his,dim=0)

    return user_features_his

def calculate_all_item_embeddings(model, meta, args):
    # preprare all item prompts
    print('****************** Computing All Item Embeddings ******************')
    candidate_prompts = []
    for item in range(1, args.num_items+1):
        candidate_text = get_target_prompt(args, item, meta)
        candidate_prompts.append(candidate_text)
    
    candidate_embeddings =[]
    
    with torch.no_grad():
        for i in tqdm(range(0, args.num_items, args.test_batch_size)):
            input_prompts = candidate_prompts[i: i + args.test_batch_size]
        
            input_tokens = model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
            input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

            outputs = model.model(**input_tokens)
            embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
            embeddings = F.normalize(embeddings, dim=-1)
            candidate_embeddings.append(embeddings)

        candidate_embeddings = torch.cat(candidate_embeddings)
        
    return candidate_embeddings

def hybrid_retrival(model_e5, model_lru, batch, meta, e5_val_user_embeddings, lru_val_user_embeddings, args):
    dp_seqs, _, dp_texts, labels = batch
    dp_seqs = dp_seqs.cuda()
    batch_size = len(dp_seqs)

    #### E5 
    input_prompts = get_batch_dp_prompt(args, dp_texts)
    input_tokens = model_e5.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
    input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

    outputs = model_e5.model(**input_tokens)
    embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
    embeddings = F.normalize(embeddings, dim=-1)

    scores_e5 = torch.matmul(embeddings, e5_val_user_embeddings.T)
    probs_e5 = torch.softmax(torch.tensor(scores_e5.clone().detach()) / 0.01, dim=-1)
    # rank_e5 = (-scores_e5).argsort(dim=1)  
    # rank_e5 = rank_e5[:, :args.topk_user]

    #### LRU
    test_user_features = model_lru.forward_feature(dp_seqs)[:, -1, :]
    scores_lru = test_user_features @ lru_val_user_embeddings.t()
    probs_lru = torch.softmax(torch.tensor(scores_lru.clone().detach()), dim=-1)
    # rank_lru = (-scores_lru).argsort(dim=1) 
    # rank_lru = rank_lru

    ensembled_probs = args.lambda_ensemble * probs_e5 + (1 - args.lambda_ensemble) * probs_lru 
    rank = (-ensembled_probs).argsort(dim=1)
    rank = rank[:, :args.topk_user]

    # pdb.set_trace()
    #### common interets
    in_context_samples = []
    for i in range(batch_size):
        candidates = rank[i].tolist()
        in_context_samples.append(candidates)

    return in_context_samples


def dp_prompt(args, dp_texts):
    prompt = "query: "
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new']:
        for item_text in dp_texts:
            title = item_text[0]
            category = item_text[1]
            category = ', '.join(category.split(', ')[-2:])

            prompt += title
            prompt += ', a product about '

            prompt += category
            prompt += '. \n'

    if args.dataset_code in ['ml-100k']:
        for item_text in dp_texts:
            title = item_text[0]
            category = item_text[1]
            category = ', '.join(category.split(', ')[-2:])

            prompt += title

            prompt += category
            prompt += '. \n'

    return prompt

def get_batch_dp_prompt(args, dp_texts):
    input_prompts = []
    num_samples = len(dp_texts)
    for i in range(num_samples):
        dp_text = dp_texts[i]
        input_text = dp_prompt(args, dp_text)
        input_prompts.append(input_text)
    return input_prompts


def get_batch_prompts(args, batch, meta):
    input_prompts = []
    for seq, answer in zip(batch[0], batch[1]):
        input_text = get_input_prompt(args, seq, meta)
        input_prompts.append(input_text)
    return input_prompts

def main(args, export_root=None):
    seed_everything(args.seed)
    client_data, test_data, meta = dataloader_factory(args)

    # E5 model
    model_e5 = E5Model()
    load_dir_e5 = os.path.join('../trained_e5', args.dataset_code, 
                                'num_clients_' + str(args.num_clients), 
                                'samples_per_client_' + str(args.num_samples),
                                'e5')
    load_name_e5 = os.path.join(load_dir_e5, 'model.checkpoint')
    print("E5 checkpoint loaded: ", load_name_e5)
    e5_checkpoint = torch.load(load_name_e5)
    model_e5.load_state_dict(e5_checkpoint['state_dict'])
    model_e5.cuda()

    # LRU model
    model_lru = LRURec(args)
    load_dir_lru = os.path.join('../trained_lru', args.dataset_code, 
                                'num_clients_' + str(args.num_clients), 
                                'samples_per_client_' + str(args.num_samples),
                                'lru')
    load_name_lru = os.path.join(load_dir_lru, 'model.checkpoint')
    print("LRU checkpoint loaded: ", load_name_lru)
    lru_checkpoint = torch.load(load_name_lru)
    model_lru.load_state_dict(lru_checkpoint['state_dict'])
    model_lru.cuda()

    dp_text_file = os.path.join('../DP_texts', args.dataset_code, 
                                'num_clients_' + str(args.num_clients), 
                                'samples_per_client_' + str(args.num_samples),
                                'dp_text', 'num_replace_' + str(args.num_replace),
                                'dp_texts.pkl')

    with open(dp_text_file, 'rb') as f:
        dp_text = pickle.load(f)['texts']

    for idx in range(args.num_clients):
        save_root = os.path.join("../IC_examples", args.dataset_code, 
                                'num_clients_' + str(args.num_clients), 
                                'samples_per_client_' + str(args.num_samples),
                                'in_context_num_replace_' + str(args.num_replace), 
                                'lambda_ensemble_' + str(args.lambda_ensemble), 
                                'client_' + str(idx))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        print("we are saving results to: ", save_root)
        
        retrieve_in_context_examples(model_e5, model_lru, client_data[idx], test_data, dp_text, meta, save_root, args)

if __name__ == "__main__":
    set_template(args)
    main(args, export_root=None)
