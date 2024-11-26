import pdb
import numpy as np
import torch
from math import e

# def build_1_order_synonym_dict(lambda_ensemble, min_cos_sim, item_embeddings_e5, item_embeddings_lru, min_items=2, max_items=25):
def build_1_order_synonym_dict(min_cos_sim, item_embeddings_e5, min_items=2, max_items=25):
    synonym_matrix = []
    synonym_dict_1_order = {}
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(len(item_embeddings_e5)):
        # item id starts from 1, 
        # but item_embeddings starts with feature id 0 for item 1
        item_id = i + 1
        item_e5 = item_embeddings_e5[i]
        scores_e5 = item_e5 @ item_embeddings_e5.t()
        probs_e5 = torch.softmax(scores_e5.unsqueeze(-1) / 0.01, dim=0).squeeze()

        # item_lru = item_embeddings_lru[i]
        # scores_lru = item_lru @ item_embeddings_lru.t()
        # probs_lru = torch.softmax(scores_lru.unsqueeze(-1), dim=0).squeeze()

        # scores = lambda_ensemble * probs_e5 + (1 - lambda_ensemble) * probs_lru 
    
        _, indices = torch.sort(scores_e5, descending=True)
        indices = indices[:(min_items+1)]

        # save the real item id 
        # not the feature id
        synonym_items = (indices+1).squeeze().cpu().numpy().tolist() 

        if item_id in synonym_items:
            synonym_items.remove(item_id)

        if len(synonym_items) > max_items:
            synonym_items = np.random.choice(synonym_items, max_items, replace=False).tolist()
            
        synonym_dict_1_order[item_id] = synonym_items
        synonym_matrix.append(synonym_items[:2])
    
    synonym_matrix.append([len(item_embeddings_e5)+1]*2)

    return synonym_matrix, synonym_dict_1_order
    
def build_2_order_synonym_dict(synonym_matrix, synonym_dict_1_order, max_items=25):
    synonym_dict_2_order = {}
    counter = 0
    for item_id in synonym_dict_1_order.keys():
        synonym_items_2_order = []
        synonyms = synonym_dict_1_order[item_id]
        for synonym in synonyms:
            synonym_items_2_order += synonym_dict_1_order[synonym]
        synonym_items_2_order = set(synonym_items_2_order)
        if item_id in synonym_items_2_order:
            synonym_items_2_order.remove(item_id)
        
        for synonym in synonyms:
            if synonym in synonym_items_2_order:
                synonym_items_2_order.remove(synonym)
        if len(synonym_items_2_order) > max_items:
            synonym_items_2_order = list(synonym_items_2_order)
            synonym_items_2_order = np.random.choice(
                synonym_items_2_order, max_items, replace=False).tolist()
        synonym_dict_2_order[item_id] = list(synonym_items_2_order)

        if len(list(synonym_items_2_order)) == 1 or len(list(synonym_items_2_order)) == 2:
            temp = list(synonym_items_2_order) + list(np.random.choice(list(synonym_items_2_order), 3 - len(list(synonym_items_2_order))))
            synonym_matrix[counter] += temp
        elif len(list(synonym_items_2_order)) == 0:
            temp = list(np.random.choice(range(1, len(synonym_matrix)+1), 3))
            synonym_matrix[counter] += temp
        else:
            synonym_matrix[counter] += list(synonym_items_2_order)[:3]
    
        counter += 1

    synonym_matrix[counter] += [len(synonym_matrix)] * 3
    synonym_matrix = torch.cat([torch.zeros(1,5), torch.Tensor(synonym_matrix)], dim=0).cuda().long()

    return synonym_matrix

def utility_scores(epsilon, x_e5, x_e5_primes):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    num_neighbors = len(x_e5_primes)
    # x = torch.cat([x_e5.unsqueeze(0), x_lru.unsqueeze(0)], dim=1).squeeze()
    # x_primes = torch.cat([x_e5_primes, x_lru_primes], dim=-1).squeeze()
    
    cos_similarity = cos(x_e5.unsqueeze(0), x_e5_primes) 
    util_scores = torch.exp(cos_similarity)
    sensitivity = e - 1
    
    util_scores = torch.exp(epsilon * util_scores / (2 * sensitivity)) / 0.01
    probs = util_scores / torch.sum(util_scores)

    return probs

def get_similar_item(x, smoothed_x, all_item_embeddings):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    x_similarity = cos(x.unsqueeze(0), all_item_embeddings)
    smoothed_x_similarity = cos(smoothed_x.unsqueeze(0), all_item_embeddings)
    original_id = torch.argmax(x_similarity, dim=-1)
    new_ids = torch.topk(smoothed_x_similarity, k=2)[1]
    if new_ids[0] == original_id:
        return new_ids[1] + 1 # convert to real item id
    else:
        return new_ids[0] + 1 # convert to real item id
    

# def dirichlet_smoothing(self, seqs, x, sensitive_pattern):
#     batch_size = x.shape[0]
#     x_smoothed = x.clone().detach()
#     counter = 0
#     for i in range(batch_size):
#         sensitive_span = sensitive_pattern[i]  
#         sensitive_seqs = seqs[i][sensitive_span[0]:sensitive_span[1]]
#         neightbors = torch.cat([sensitive_seqs.unsqueeze(1), self.synonym_matrix[sensitive_seqs]], dim=-1).reshape(1,-1)[0]
        
#         neightbor_num = sensitive_span[1] - sensitive_span[0]

#         neightbors_embedding = self.model.embedding.token.weight.detach()[neightbors].reshape(neightbor_num, len(neightbors) // neightbor_num, -1)
        

#         dirichlet_weights_temp = dirichlet_sampling(len(neightbors) // neightbor_num)
#         smoothed_temp = torch.Tensor(dirichlet_weights_temp).unsqueeze(0).unsqueeze(-1).to(self.device)

#         seq_smoothed = torch.sum(neightbors_embedding * smoothed_temp, dim=1)
        
#         x_smoothed[i, sensitive_span[0]:sensitive_span[1], :64] = seq_smoothed

#     return x_smoothed

