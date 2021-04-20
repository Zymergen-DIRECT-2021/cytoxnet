# Cytoxnet Use Cases
The package is intended as an exploration and a toolbox for the prediction of
cytotoxicity of compounds, but aims to leverage datasets of other targets
for transfer learning.
***
***

1. <span style="color:blue">Load datasets into python uniformly</span>
    - *Loading datasets into memory, whether from disk or one of the package
      options.*
    - _User_: Provides the name of a dataset, or a file on disk. Additionally
      for files the target columns, any columns to consider for dropping
      duplicates and Nans.
    - _Returns_: Cleaned dataset (unrelevant info dropped, Nans and duplicates
      handled)
    
2. <span style="color:blue">Featurize and prepare dataset (@deepchem)</span>
    - *Featurizing previously unreadable inputs such as 2D structure stings eg.
      SMILES or InChI.*
    - _User_: Provides a dataset and a pointer towards any structure to expand
    - _Returns_: The dataset with featurization appended.
    
3. <span style="color:blue">Preprocessing (@deepchem)</span>
    - *Converting dataset to machine readable format and performing any
      transformations used to improve model performance while encoding chemical
      intuition, eg. Data normalization, log tranformations*
    - _User_: Provides a dataset a list of transformation names to undertake
    - _Returns_: The dataset ready for machine reading.
    
4. <span style="color:blue">Produce toxicity models (@deepchem)</span>
    - - *Getting help of available models.*
      - _User_: Provides model name.
      - _Returns_: Help on target model or list of available models.
    - - *Initialization of model, access many types.*
      - _User_: Provides a model type name to use, and hyperparameters.
      - _Returns_: Instance of model desired architecture with hyperparameters.
    - - *Training of the model.*
      - _User_: Provides a model and a ML ready set of data to train on.
      - _Returns_: Trained model and any metrics associated with training.
    - - *Using the model to predict.*
      - _User_: Provides a model and a set of data prepared in the same way
        as the training data to predict on.
      - _Returns_: Predictions on target of data.
    - - *Scoring the model on testing data.*
      - _User_: Provides a model and a set of data prepared in the same way
        as the training data to use for scoring.
      - _Returns_: Predictions on testing of data and testing metrics.
    
5. <span style="color:blue">Optimize hyperparameters of model (@deepchem)</span>
    - *Search for optimum hyperparameters for a model class using training and
      validation data.*
    - _User_: Provides a model class name, hyperparameters to search for, a
      search scheme, and training/validation datasets.
    - _Returns_: The dataset ready for machine reading.
    
6. <span style="color:blue">Save and load pretrained models (@deepchem)</span>
    - *Using old models and saving new ones for later. Different model types
    require different handling.*
    - _User_: Provides a model and a place to put it, or the location of disk of
    the old model.
    - _Returns_: Save model to disk or load old one.
    
7. <span style="color:blue">Transfer to new target/species</span>
    - *High level transfer is simply continueing to train on different data. 
      Provide options to also fix models to allow for more beneficial
      pretraining on larger datasets.*
    - _User_: Provides a trained model and a new training dataset, and any
      transfer options.
    - _Returns_: Retrained model.
    
8. <span style="color:blue">Visual analysis of model</span>
    - *Provide visual evaluation of a model performance in conjunction with base
      metrics.*
    - _User_: Provides a model and a set of data prepared in the same way
      as the training data to use for scoring.
    - _Returns_: Metrics and Visuals
    
9. <span style="color:blue">Evaluate transferability of datasets</span>
    - *Evaluates overlap of datasets. *
    - _User_: Provides two datasets with uniform featurization.
    - _Returns_: Metrics and Visuals