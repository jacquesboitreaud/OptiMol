# graph2smiles

Molecular graph to SELFIES VAE. 

Required : 
pytorch, dgl 
pandas 
moses to get the data (pip install molsets)
selfies (pip install selfies)
rdkit
tqdm

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
The column containing the smiles should be labeled 'smiles'. 

Arguments : 
- -i : path to dataset of molecules to embed
- -v : 'smiles' or 'selfies', the type of output the model was trained for 
- -m : path to .pth file containing trained model weights (default is 'saved_model_w/baseline.pth')
- -d : Optional argument without input value, store true to decode the latent points into smiles

```
python embed_mols.py -i [my_dataset.csv] -v [output_type]
```

### Generating samples (TODO)

To generate N samples from the trained model, run : 
smiles of molecules generated will be written to text file 'data/gen.txt' if no -o arg specified. 
```
python generate/sample_prior.py -n N -o [my_output_file]
```

