import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import wandb
import argparse

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def main(args, export_root=None):
    seed_everything(args.seed)
    client_data, test_data, meta = dataloader_factory(args)
    model = E5Model()
    if export_root == None:
        export_root = os.path.join(EXPERIMENT_ROOT, args.dataset_code, 
                                    'num_clients_' + str(args.num_clients), 
                                    'samples_per_client_' + str(args.num_samples),
                                    'e5')
    
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    print("we are saving results to: ", export_root)

    model.cuda()

    # copy weights
    global_weights = model.state_dict()

    # Training
    for epoch in range(args.global_epochs):
        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        model.train()
        for idx in range(args.num_clients):
            local_model = E5Trainer(args=args, model=copy.deepcopy(model), client_data=client_data[idx], client_id=idx, global_round=epoch, meta=meta, 
                                    E5TrainDataset=E5TrainDataset, E5ValidDataset=E5ValidDataset, collate_fn=collate_fn)
            
            w = local_model.train()
            
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        model.load_state_dict(global_weights)

    model_save_name = os.path.join(export_root, 'model.checkpoint')
    model_checkpoint = {'state_dict': model.state_dict()}
    torch.save(model_checkpoint, model_save_name)


if __name__ == "__main__":
    set_template(args)
    main(args, export_root=None)
