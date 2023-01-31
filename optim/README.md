# Bayesian optimization in chemical space :

We use Bayesian Optimization in latent space as a baseline molecular optimization process.
To reproduce the Bayesian Optimization benchmarks on clogP optimization :

unzip the 250k zinc molecules dataset in /data : unzip 250k_zinc.zip

### Steps to reproduce BO benchmark on cLogP

Compute clogP for random samples in the 250k dataset by running 

```
python generate_init.py 
```

Run 10 sochastic runs of BO with 50 new samples per step (same params as Kusner et al. and Jin et al.) with 
```
python run_bo.py --name benchmark
```

Parse the results : 
```
python parse_results.py --name benchmark --n_sim 10 --n_iters 20
```

### To run BO with bigger batches and compare to OptiMol:

Compute clogP for random samples in the 250k dataset by running 
```
python generate_init.py 
```

Run 1 run of BO with 500 new samples per step and 20 steps (~ 24 hours running time): 
```
python run_bo.py  --name big_run --bo_batch_size 500 --n_init 10000
```

Parse the results : 
```
python parse_results.py --name big_run --n_sim 1 --n_iters 20
```