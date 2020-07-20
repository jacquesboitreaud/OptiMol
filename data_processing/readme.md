### Data processing utilities 

Parse dataset to get all bond types and build one-hot edge features : 

```
get_edge_data.py -csv [my_csv_dataset]
```

Compute chemical properties and add them as columns to csv dataset : 
```
chem_props.py -csv [my_csv_dataset]
```