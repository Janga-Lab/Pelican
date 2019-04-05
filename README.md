# Pelican
This Project Predics m1A modification sites on Direct RNA sequencing

### Getting a Git Repository of pelican
    git clone https://github.iu.edu/pkothiya/Pelican


### How to run Pelican

#### Prerequisites
Create Conda Enviroment in python 3. if you need help --> https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Load following packages to enviroment

    itertools
    h5py
    numpy
    re
    os
    argparse
    keras
    sklearn
    tensorflow
    
Or use requirements.txt to install all above packages
    
    pip install -r pelican/requirements.txt    

#### Download model file
    wget https://deeplearn.soic.iupui.edu/~pkothiya/0.6_prob_keras_m1A_v2.h5
    
Note: Your model file "0.6_prob_keras_m1A_v2.h5" should be in working directory 

#### Run Pelican
    python path/to/Pelican/pelican.py -i /workspace/pass/ -b 200

#### Available options

-i fast5 file location

-n Provide nucleotide base type to predict modifications(ex:A)--> Default: A (Do Not use this option in current version)

-b Process files in batches depending on your available resources..smaller batch size is recommended for less computational resources
 


