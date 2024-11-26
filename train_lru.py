import os
import torch
import pdb

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pickle
import wandb
import argparse

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

def average_weights(weights):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))
    
    return w_avg

def generate_candidates(model, test_data, retrieved_data_path, args):
    # prepare test dataloader
    test_dataset = LRUTestDataset(args, test_data, args.bert_max_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)

    model.eval()
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    committe = {}
    with torch.no_grad():
        print('****************** Generating Candidates for Test Set ******************')
        tqdm_dataloader = tqdm(test_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.cuda() for x in batch]
            seqs, labels = batch
    
            scores = model(seqs)[:, -1, :]
            scores[:, 0] = -1e9  # padding
            
            test_probs.extend(scores.tolist())
            test_labels.extend(labels.view(-1).tolist())
        test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
                                                    torch.tensor(test_labels).view(-1), args.metric_ks)
        print(test_metrics)

    with open(retrieved_data_path, 'wb') as f:
        pickle.dump({'test_probs': test_probs,
                    'test_labels': test_labels,
                    'test_metrics': test_metrics}, f)

def main(args, export_root=None):
    seed_everything(args.seed)
    client_data, test_data, _ = dataloader_factory(args)

    # Training
    for idx in range(args.num_clients):
        export_root = os.path.join(EXPERIMENT_ROOT, args.dataset_code, 
                                    'num_clients_' + str(args.num_clients), 
                                    'samples_per_client_' + str(args.num_samples),
                                    'lru', 'client_' + str(idx))
        
        if not os.path.exists(export_root):
            os.makedirs(export_root)
        print("we are saving results to: ", export_root)

        model = LRURec(args)
        model.cuda()
        model.train()

        local_model = LRUTrainer(args=args, model=copy.deepcopy(model), 
                                client_data=client_data[idx], client_id=idx)
        
        w = local_model.train()
        
        model.load_state_dict(w)
        generate_candidates(model, test_data, os.path.join(export_root, 'retrieved.pkl'), args)
        
        model_save_name = os.path.join(export_root, 'model.checkpoint')
        model_checkpoint = {'state_dict': w}
        torch.save(model_checkpoint, model_save_name)


if __name__ == "__main__":
    args.model_code = 'lru'
    set_template(args)
    main(args, export_root=None)