# graph2smiles

Molecular graph to SELFIES VAE. 

Required : 
pytorch, dgl 
pandas 
moses to get the data (pip install molsets)
selfies (pip install selfies)

### Data loading

We use Molecular Sets data to train our model : https://github.com/molecularsets/moses 

After installing the moses python library, the data can be reached by running 

```
python data/download_moses.py 
```

### Model training 

To retrain the model on the moses train set, with default settings, run
```
python train.py
```

To train the model on your own dataset (csv file), run 
```
python train.py --train [my_dataset.csv]
```

### Embedding molecules 

To compute embeddings for molecules in csv file, run 
If csv has several columns, the column containing the smiles should be labeled 'smiles'. 

```
python embed_mols.py -i [my_dataset.csv]
```

### Generating samples (TODO)

To generate N samples from the trained model, run : 
```
python sample.py -n N
```

