# OptiMol

This is the code for the paper on https://www.biorxiv.org/content/10.1101/2020.05.23.112201v2

This repo introduces two things :

- A new Variational Auto-Encoder (VAE) architecture that goes from a molecular
graph to a sequence representation (and especially SELFIEs).
- An optimization pipeline to iteratively shift a prior distribution to maximize a black-box scoring function.
In our implemntation, this back-box function is the score returned by a docking software.

The necessary packages are packaged as yml files available for cpu or cuda10 usage.

 ```
conda env create -f ymls/cpu.yml 
```

## Prior model training: learning a distribution in molecules space

#### Data loading

We use Molecular Sets (https://github.com/molecularsets/moses) 
to train our model : 
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
python generate/sample_prior.py -N [number_of_samples] --name [name_of_the_model] -b [use beam search decoding]
```

#### Moses metrics 

To compute the Moses benchmark metrics for the samples (recommended 30k samples), run 
```
python eval/moses_metrics.py -i [path_to_txt_with_samples]
```

## OptiMol: Generating samples that maximize a black-box objective function : 

- The Bayesian Optimization baseline : `/optim`
- Our implementation of Conditioning By Adaptive Sampling (https://arxiv.org/abs/1901.10060): `/cbas`



