# ec-code
Code for the analysis of eurydendroid cells data.

Code is being moved here from [this private repo](https://github.com/portugueslab/eurydendroid-analysis).

Code that has been moved:

 - [x] E0050 with responses to motor activity w/ and w/o visual feedback for ECs, PCs, IONs, GCs
 - [ ] E0022 responses of ECs to different visual stimuli
 - [ ] Analysis of ephys data from EC recordings 


### Specify dataset location
To handle compatibility with local vs. FUNES vs. otherwise stored datasets, 
the master folder that contains all data should be specified in a `dataset_location.txt`
file that contains a single string with the folder, placed in the repo (ec-code/dataset_location.txt).
This file is git-ignored (with the .gitignore file) as it has to remain local.

### Installation with notebook clean hook

To commit notebooks without content, install `nb-clean`:
``` 
python3 -m pip install nb-clean
```

And in the repo local path:
```
nb-clean add-filter
```
