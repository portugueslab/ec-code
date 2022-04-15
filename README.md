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

## Description of the analysis
Here is some general guideline for how to run the analyses for the different projects.

###Visual feedback effect - E0050 (`fb_effect`)

In this analysis, scripts are in `scripts/fb_effect`, notebooks in `scripts/fb_effect`,
and utilities in `ec_code/fb_effect`. The overall concept is to work with information aggregated from all fish in dataframes:
 - `exp_df`: contains info for every experiment (genotype, quality, fish adaptation, etc.)
 - `cells_df`: contains info for every cell from every fish (coordinates, fish, ...); when then performing motor- and sensory- related analysis, additional columns are added with info on motor/sensory responsiveness, etc.
 - `bouts_df`: info for every individual bout from all fish (time, bout stats, fish, etc,)
 - `trial_df`: info for every individual trial from all fish (time, speed, bouts occorred, timing of the bout, fish, etc.)
 - `traces_df`: huge matrix with traces from all cells, with column names corresponding to unique cell indexes

Steps:

- data from all individual fish is pooled together in huge flammkuchen 
files (using `0_create_dataframes.py`)
- some preprocessing on bouts is computed, and dataframes that pool together info for every experiment, bout, trial, cell are created (using `1_preprocess_dataframes.py`)
