EEGDetectionWithMetaplasticity
==============================

This is the git repo associated with the article _Metaplastic-EEG: Continuous training on brain-signals_ by

Thomas Bersani--Veroni, Isabelle Aguilar (Electronic mail: iagu0459@sydney.edu.au.), Luis Fernando Herbozo Contreras, Armin Nikpour, and Omid Kavehei

School of Biomedical Engineering, The University of Sydney, NSW 2006, Australia.


# Project Organization
------------

    
    ├── README.md          <- This file
    │
    ├── reports            
    │   └── ArticleFigures.ipynb        <- Generated graphics and figures used in the Article
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment on linux.
    ├── requirements_windows.txt   <- The requirements file for reproducing the analysis environment on windows.
    │
    ├── src                
    │   ├── data           
    │   │   ├── TUHDEdfToPckl.py <- Script to convert TUH EEG .edf to 12 sec long pckl files.
    │   │   └── TUHPcklToNpyMultiprocess.py <- Script to convert TUH EEG pckl files to two x and y npy array used for training
    │   │
    │   ├── models      
    │   │   └──StreamTUH.py <- script to train a BNN on a stream of seizure data   
    │   │   └──SyntheticTaskTUH.py <- script to train a BNN on synthetic datasets for seizure detection.



--------
# Usage

## To download the repository and install its dependencies 

On linux : 
```terminal
git clone [!!!]
cd EEGDETECTIONWITHMETAPLASTICITY
python -m venv venv --prompt="MetaplasticEEG"
./venv/Scripts/activate
pip install -r requirements.txt
```

On windows:

```
git clone [!!!]
cd EEGDETECTIONTWITHMETAPLASTICITY
python -m venv venv --prompt="MetaplasticEEG"
./venv/Scripts/activate
pip install -r requirements_windows.txt
```
## Then to preprocess TUH's raw data

First execute ```TUHEdfToPckl.py``` while changing the data folder in/out in the script. Then do the same with ```TUHPcklNpyMultiprocess.py```.

## To run the experiments

Execute the script associated with the experiments you want to run. The hyperparameters can be changed at the top of each file beside the ```#Hyperparameter``` comment. Theses script use wandb to track the logs of the training.



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
