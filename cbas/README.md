## Molecular optimization

This tutorial enables using our tool to optimize certain properties.
The method is flexible in the function that we are trying to optimize, and was 
designed to compute properties that are long to get on a cpu. 
The default setting uses VINA for docking on drd3. However it is very easy
to adapt it to another receptor, just by changing the conf.txt and .pdbqt
 that is saved in the root/docking/data_docking/ folder.

To adapt OptiMol to other scoring functions or other docking tools, some coding
is needed, but we designed the code for a maximum flexibility.
One should implement an [EXAMPLE] function following the template 

```python
def [EXAMPLE](smiles_list : List[str], name: str, unique_id: str) -> None :
    dirname = os.path.join(script_dir, 'results', name, 'docking_small_results')
    dump_path = os.path.join(dirname, f"{unique_id}.csv")
    """
    do your calculations on the list here and dump results 
    as a text file with one smile and one score per row,
    at the dump_path address
    """
```

and then add it as a switch in the docker.py main() function 

```python
elif oracle == '[EXAMPLE]':
    [EXAMPLE](smiles_list,
              name=name,
              unique_id=proc_id)
``` 

We have implemented a multi-core version for one node, but also one 
 that can leverage a distributed network equipped with SLURM task manager.


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

For instance if [EXAMPLE] was trained over [example_prior] and one wants to use
the model obtained after [10] iterations, these lines will make the model usable 

```
mkdir ../results/saved_models/[EXAMPLE]
cp results/[EXAMPLE]/weigths_[10].pth ../results/saved_models/[EXAMPLE]/weights.pth
cp ../results/saved_models/[example_prior]/params.json ../results/saved_models/[EXAMPLE]/
```

And then one can use the model 'example' from the root as explained in the root README.md


