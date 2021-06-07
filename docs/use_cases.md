# Cytoxnet Use Cases
The package is intended as an exploration and a toolbox for the prediction of
cytotoxicity of compounds, but aims to leverage datasets of other targets
for transfer learning.
***
***

1. <span style="color:blue">Load datasets into python uniformly</span>
    - *Loading datasets into memory, whether from disk or one of the package
      options.*
    - __User__: Provides the name of a dataset, or a file on disk. Additionally
      for files the target columns, any columns to consider for dropping
      duplicates and Nans.
    - __Returns__: Cleaned dataset (unrelevant info dropped, Nans and duplicates
      handled)
    
2. <span style="color:blue">Featurize and prepare dataset (@deepchem)</span>
    - *Featurizing previously unreadable inputs such as 2D structure stings eg.
      SMILES or InChI.*
    - __User__: Provides a dataset and a pointer towards any structure to expand
    - __Returns__: The dataset with featurization appended.
    
3. <span style="color:blue">Preprocessing (@deepchem)</span>
    - *Converting dataset to machine readable format and performing any
      transformations used to improve model performance while encoding chemical
      intuition, eg. Data normalization, log tranformations*
    - __User__: Provides a dataset a list of transformation names to undertake
    - __Returns__: The dataset ready for machine reading.
    
4. <span style="color:blue">Produce toxicity models (@deepchem)</span>
    - - *Getting help of available models.*
      - __User__: Provides model name.
      - __Returns__: Help on target model or list of available models.
    - - *Initialization of model, access many types.*
      - __User__: Provides a model type name to use, and hyperparameters.
      - __Returns__: Instance of model desired architecture with hyperparameters.
    - - *Training of the model.*
      - __User__: Provides a model and a ML ready set of data to train on.
      - __Returns__: Trained model and any metrics associated with training.
    - - *Using the model to predict.*
      - __User__: Provides a model and a set of data prepared in the same way
        as the training data to predict on.
      - __Returns__: Predictions on target of data.
    - - *Scoring the model on testing data.*
      - __User__: Provides a model and a set of data prepared in the same way
        as the training data to use for scoring.
      - __Returns__: Predictions on testing of data and testing metrics.
    
5. <span style="color:blue">Optimize hyperparameters of model (@deepchem)</span>
    - *Search for optimum hyperparameters for a model class using training and
      validation data.*
    - __User__: Provides a model class name, hyperparameters to search for, a
      search scheme, and training/validation datasets.
    - __Returns__: The dataset ready for machine reading.
    
6. <span style="color:blue">Save and load pretrained models (@deepchem)</span>
    - *Using old models and saving new ones for later. Different model types
    require different handling.*
    - __User__: Provides a model and a place to put it, or the location of disk of
    the old model.
    - __Returns__: Save model to disk or load old one.
    
7. <span style="color:blue">Transfer to new target/species</span>
    - *High level transfer is simply continueing to train on different data. 
      Provide options to also fix models to allow for more beneficial
      pretraining on larger datasets.*
    - __User__: Provides a trained model and a new training dataset, and any
      transfer options.
    - __Returns__: Retrained model.
    
8. <span style="color:blue">Visual analysis of model</span>
    - *Provide visual evaluation of a model performance in conjunction with base
      metrics.*
    - __User__: Provides a model and a set of data prepared in the same way
      as the training data to use for scoring.
    - __Returns__: Metrics and Visuals
    
9. <span style="color:blue">Evaluate transferability of datasets</span>
    - *Evaluates overlap of datasets.*
    - __User__: Provides two datasets with uniform featurization.
    - __Returns__: Metrics and Visuals

10. <span style="color:blue">Manage resource pool</span>
    - *Produce and update a pool of sparse data to use for augmentation.*
    - __User__: Provides datasets or specifies package data, and features to use.
    - __Returns__: None, resource pool is updated.

11. <span style="color:blue">Augment dataset</span>
    - *Augment a small user dataset with data from the resource pool.*
    - __User__: Provides a dataset of compounds and desired targets.
    - __Returns__: Dataset with new targets and compounds added.

12. <span style="color:blue">Build and manage data within database schema</span>
    - *Create and automatically update relational database of package and user data.*
    - __User__: Provides new datasets with compounds, targets, and activity values.
    - __Returns__: Relational database with tables for each dataset added.
