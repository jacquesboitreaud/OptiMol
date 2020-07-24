# OptiMol

This is the code for the paper on https://www.biorxiv.org/content/10.1101/2020.05.23.112201v2

This repo introduces two things :

- A new Variational Auto-Encoder (VAE) architecture that goes from a molecular
graph to a sequence representation (and especially SELFIEs).
- An optimization pipeline that optimizes a scoring function that includes 
docking

The necessary packages are packaged as ymls available for cpu or cuda10 usage.

 ```
conda env create -f ymls/cpu.yml 
```

Otherwise one should manually install the following packages :

pytorch, dgl, networkx, scikit-learn,rdkit, tqdm, ordered-sets, moses, pandas

## Prior model training

#### Data loading

We use Molecular Sets (https://github.com/molecularsets/moses) 
to train our model : 
As a one step command, one can run 
After installing the moses python library, the data can be reached by running 

```
python data_processing/download_moses.py 
```
To train a graph2selfies model, selfies need to be precomputed for the train set by running 
To compute selfies for another dataset stored in csv, the molecules should be
 in a column entitled 'smiles', run : 
```
python data_processing/get_selfies.py -i [path_to_my_csv_dataset]
```

#### Model training 

To train the model run 
```
python train.py --train [my_dataset.csv] --n [your_model_name]
```
The csv must contain columns entitled 'smiles' and 'selfies'

#### Embedding molecules 

To compute embeddings for molecules in csv file:
```
python embed_mols.py -i [path_to_csv] --name [your_model_name] -v [smiles]/[selfies]
```
The column containing the smiles/selfies should be labeled 'smiles'. 

#### Generating samples

To generate samples from a trained model, run : 
```
python generate/sample_prior.py -N [number_of_samples] --name [name_of_the_model]
```

#### Moses metrics 

To compute the Moses benchmark metrics for the samples (recommended 30k samples), run 
```
python eval/moses_metrics.py -i [path_to_txt_with_samples]
```

## Scoring function optimization


This is mostly an efficient implementation of the CbAS algorithm for docking.
there is also two implementations for BO in /optim

#### OptiMol

Go to /cbas
 



