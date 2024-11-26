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


def generate_dp_texts(model, test_data, meta, retrieved_data_path, args):
    # prepare test dataloader
    test_dataset = E5TestDataset(args, test_data, args.bert_max_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                                collate_fn=collate_fn)

    model.eval()
    test_dp_texts = []
    with torch.no_grad():
        candidate_embeddings_e5 = calculate_all_item_embeddings(model, meta, args)
        synonym_matrix, synonym_dict_1_order = build_1_order_synonym_dict(args.min_cos_sim, candidate_embeddings_e5)
        synonym_matrix = build_2_order_synonym_dict(synonym_matrix, synonym_dict_1_order)
        
        tqdm_dataloader = tqdm(test_loader)
        print('****************** Generating Candidates for Test Set ******************')
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = len(batch[0])
            # new_seqs = calculate_substitutions(batch, synonym_matrix, candidate_embeddings_e5, candidate_embeddings_lru, args)
            new_seqs = calculate_substitutions(batch, synonym_matrix, candidate_embeddings_e5, args)
            for i in range(batch_size):
                seq = new_seqs[i]
                dp_text = []
                for item in seq:
                    dp_text.append([meta[item][0], meta[item][2]])
                test_dp_texts.append(dp_text)

    with open(retrieved_data_path, 'wb') as f:
        pickle.dump({'texts': test_dp_texts}, f)


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


def calculate_substitutions(batch, synonym_matrix, all_item_embeddings_e5, args):
    seqs, labels = batch
    batch_size = len(seqs)
    new_seqs = []
    # pdb.set_trace()
    for i in range(batch_size):
        seq = copy.deepcopy(seqs[i])

        seq_len = len(seq)
        try:
            replace_pos = np.random.choice(seq_len-1, args.num_replace, replace=False)
        except:
            replace_pos = np.random.choice(seq_len-1, seq_len-1, replace=False)

        for pos in replace_pos:
            synonym_set = synonym_matrix[seq[pos]]
            if len(all_item_embeddings_e5)+1 in synonym_set:
                synonym_set_list = synonym_set.tolist()
                synonym_set_list.remove(len(all_item_embeddings_e5)+1)
                synonym_set = torch.tensor(synonym_set_list).long().cuda()

            neighbor_embeddings_e5 = all_item_embeddings_e5[synonym_set-1]
            # neighbor_embeddings_lru = all_item_embeddings_lru[synonym_set-1]

            dp_probs = utility_scores(args.dp_epsilon, all_item_embeddings_e5[seq[pos]-1], neighbor_embeddings_e5)

            smoothed_embedding = torch.sum(neighbor_embeddings_e5 * dp_probs.unsqueeze(1), dim=0)
            dp_synonym = get_similar_item(all_item_embeddings_e5[seq[pos]-1], smoothed_embedding, all_item_embeddings_e5)
            seq[pos] = dp_synonym.item()
            
        new_seqs.append(seq)

    return new_seqs

def calculate_logits(model, new_seqs, batch, meta, candidate_embeddings, args):
    _, labels = batch
    batch_size = len(new_seqs)

    input_prompts = get_batch_prompts(args, meta, new_seqs, labels)
    batch_size = len(input_prompts)

    input_tokens = model.tokenizer(input_prompts, max_length=256, truncation=True, padding=True, return_tensors="pt")
    input_tokens = {k: v.cuda() for k, v in input_tokens.items()}

    # forward pass
    outputs = model.model(**input_tokens)
    embeddings = average_pool(outputs.last_hidden_state, input_tokens['attention_mask'])
    embeddings = F.normalize(embeddings, dim=-1)

    # seqs, labels = batch
    scores = torch.matmul(embeddings, candidate_embeddings.T)
    # 0 itme padding
    place_holder = torch.zeros((batch_size, 1)).cuda()
    scores = torch.cat([place_holder, scores], dim=-1)
    
    for i in range(batch_size):
        scores[i, 0] = -1e9  # padding

    return scores, labels

def get_batch_prompts(args, meta, seqs, labels):
    
    input_prompts = []
    target_prompts = []
    num_samples = len(seqs)
    for i in range(num_samples):
        seq = seqs[i]
        answer = labels[i]
        input_text = get_input_prompt(args, seq, meta)
        target_text = get_target_prompt(args, answer, meta)

        input_prompts.append(input_text)
        target_prompts.append(target_text)

    return input_prompts

def main(args, export_root=None):
    seed_everything(args.seed)
    client_data, test_data, meta = dataloader_factory(args)
    model = E5Model()

    load_dir = os.path.join('../trained_e5', args.dataset_code, 
                            'num_clients_' + str(args.num_clients), 
                            'samples_per_client_' + str(args.num_samples),
                            'e5')

    load_name = os.path.join(load_dir, 'model.checkpoint')

    print("we are loading model from: ", load_name)
    model_checkpoint = torch.load(load_name)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.cuda()

    if export_root == None:
        export_root = os.path.join('../DP_texts', args.dataset_code, 
                                    'num_clients_' + str(args.num_clients), 
                                    'samples_per_client_' + str(args.num_samples),
                                    'dp_text', 'num_replace_' + str(args.num_replace))
    
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    print("we are saving dp text to: ", export_root)
    
    generate_dp_texts(model, test_data, meta, os.path.join(export_root, 'dp_texts.pkl'), args)

if __name__ == "__main__":
    set_template(args)
    args.model_code = 'e5'
    main(args, export_root=None)
