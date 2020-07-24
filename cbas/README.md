# cbas4docking
Conditioning by Adaptive Sampling for lead generation

Implementation of CbAS by David H. Brookes, Hahnbeom Park and Jennifer Listgarten: https://arxiv.org/pdf/1901.10060.pdf 
Application to docking scores optimization of small molecules. 

#### Docking

To use docking, one needs to install two programs : 
- pythonsh from MGLTools
- Vina for docking

Then these two options should be manually registered by providing the 
absolute path to their installation folder. The code should be manually modified
in the Optimol/docking/docking.py line 31 (follow the template) and add a 
computer name 'computer_name'

Then one can use the default DRD3 target, or replace with ones' target files
in Optimol/docking/data_docking. One needs to add the receptor pdbqt file 
and a conf.txt with the docking box.

#### Optimization

There are two options :
running the program on one node or running it on a cluster equiped with a slurm
task manager. One can use a custom prior by adding its name

```
# one-node implementation
python main_cbas --oracle docking --server [computer_name] --prior_name [prior_name] --name [name]
# slurm implementation
python slurm_master --oracle docking --server [computer_name] --prior_name [prior_name] --name [name]
```

This will create a 'results/name' dir with all outputs. To use a slurm script,
one should add the specifications of the nodes one wants to use following
the template provided in slurm_master.py

#### Using the finetuned models
'results/name' dir' will contain the intermediate csvs produced as well as all 
the intermediate models. To use these models, one should copy the selected weights
into OptiMol/results/saved_models/[your_name] and the .json file of the prior
used for finetuning. Then it can be used as any other prior.

For instance if [example] was trained over [example_prior] and one wants to use
the model obtained after 10 iterations, these lines will make the model usable 

```
mkdir ../results/saved_models/example
cp results/weigths_10.pth ../results/saved_models/example/weights.pth
cp ../results/saved_models/example_prior/params.json ../results/saved_models/example/
```

And then one can use the model 'example' from the root as explained in the root README.md


