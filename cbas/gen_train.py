# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:46:45 2020

@author: jacqu

Function that trains a model on samples x_i, weights w_i and returns the decoder parameters phi. 
"""

import os
import sys

import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils.clip_grad as clip
from torch.utils.data import DataLoader
from selfies import decoder

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from loss_func import CbASLoss
from dataloaders.simple_loader import SimpleDataset, collate_block
from utils import soft_mkdir
from dgl_utils import send_graph_to_device


class GenTrain():
    """ 
    Wrapper for search model iterative training in CbAS
    """

    def __init__(self, model, alphabet_name, savepath, epochs, device, lr, clip_grad, beta, processes=8, DEBUG=False,
                 optimizer='adam', scheduler='elr'):
        super(GenTrain, self).__init__()

        self.model = model
        self.json_alphabet_name = alphabet_name
        self.savepath = savepath
        soft_mkdir(self.savepath)  # create dir to save the search model
        self.device = device
        self.model.to(self.device)
        self.n_epochs = epochs

        self.processes = processes
        self.debug = DEBUG

        self.teacher_forcing = 1.0
        self.beta = beta
        # Optimizer 
        self.lr0 = lr
        self.anneal_rate = 0.9
        self.anneal_iter = 40000
        self.clip_grads = clip_grad
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr0)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr0)

        if scheduler == 'elr':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.anneal_rate)
        else:
            self.scheduler = None

        self.load_optim()

        # loader
        map_path = os.path.join(script_dir, '..', 'map_files')
        self.dataset = SimpleDataset(maps_path=map_path, vocab='selfies', alphabet=self.json_alphabet_name,
                                     debug=self.debug)

    def step(self, input_type, x, w):
        """ 
        Trains the model for n_epochs on samples x, weighted by w 
        input type : 'selfies' or 'smiles', for dataloader (validity checks and format conversions are different)
        """

        if input_type == 'smiles':
            self.dataset.pass_smiles_list(x, w)
        elif input_type == 'selfies':
            self.dataset.pass_selfies_list(x, w)

        train_loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=32,
                                  num_workers=self.processes, collate_fn=collate_block, drop_last=True)
        # Training loop
        total_steps = 0
        for epoch in range(self.n_epochs):
            print(f'Starting epoch {epoch}')
            self.model.train()

            for batch_idx, (graph, smiles, w_i) in enumerate(train_loader):
                total_steps += 1  # count training steps

                smiles = smiles.to(self.device)
                graph = send_graph_to_device(graph, self.device)
                w_i = w_i.to(self.device)

                # Forward pass
                mu, logv, z, out_smi, out_p, out_a = self.model(graph, smiles, tf=self.teacher_forcing)  # no tf
                # plot_kde(z.cpu().detach().numpy())

                # Compute CbAS loss with samples weights 
                loss = CbASLoss(out_smi, smiles, mu, logv, w_i, self.beta)
                if batch_idx == 0:
                    _, out_chars = torch.max(out_smi.detach(), dim=1)
                    print(f'CbAS Loss at batch 0 : {loss.item()}')

                    differences = 1. - torch.abs(out_chars - smiles)
                    differences = torch.clamp(differences, min=0., max=1.).double()
                    quality = 100. * torch.mean(differences)
                    quality = quality.detach().cpu()
                    print('fraction of correct characters at reconstruction : ', quality.item())

                self.optimizer.zero_grad()
                loss.backward()
                clip.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                del loss
                self.optimizer.step()

                # Annealing KL and LR
                if total_steps % self.anneal_iter == 0:
                    self.scheduler.step()
                    print("learning rate: %.6f" % self.scheduler.get_lr()[0])

                if batch_idx == 0 and self.debug:
                    smiles = self.model.probas_to_smiles(out_smi)
                    smiles = [decoder(s) for s in smiles]
                    print(smiles[:5])

        # Update weights at 'save_model_weights' : 
        print(f'Finished training after {total_steps} optimizer steps. Saving search model weights')
        self.model.cpu()
        self.dump()
        self.model.to(self.device)

    def dump(self):
        """
        Dumping and loading is a bit weird, as the models parameters are in the prior definition
        :return:
        """
        weights_path = os.path.join(self.savepath, "weights.pth")
        optim_path = os.path.join(self.savepath, "optim.pth")
        torch.save(self.model.state_dict(), weights_path)
        try:
            checkpoint = {'optimizer_state_dict': self.optimizer.state_dict(),
                          'scheduler_state_dict': self.scheduler.state_dict()}
            torch.save(checkpoint, optim_path)
        except:
            print('Optim and scheduler could not be saved')

    def load_optim(self):
        """
        If first creation, this won't load anything
        :return:
        """
        optim_path = os.path.join(self.savepath, "optim.pth")
        try:
            checkpoint = torch.load(optim_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        except FileNotFoundError:
            print('No optimizer state was found')
        #     If it is not defined, for instance when using no scheduler
        except AttributeError:
            pass

# def load_from_dir(dir):
#     dumper = Dumper()
#     params = dumper.load(os.path.join(dir, 'params_gentrain.json'))
#     model = Model(**params)
#     if load_weights:
#         try:
#             model.load(os.path.join(dir, "weights.pth"))
#         except:
#             print('Weights could not be loaded by the util functions')
#     return model
