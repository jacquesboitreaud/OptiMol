generate_around.py : script to sample around a molecule in latent space 
usage : 
generate_around.py -m [smiles of seed compound] -N [number of molecules to sample] -o [name of output file]

generate_prior.py: script to generate molecules from normal prior. 
usage : 
generate_around.py -m [smiles of seed compound] -N [number of molecules to sample] -o [name of output file]

novelty.py : 
functions to assess novelty of generated compounds;
- filter those who appear in training sets 
- filter with undesirable properties / weird molecules (long carbon chain etc...)
- tanimoto similarity to retrieve most similar compounds in given set 

