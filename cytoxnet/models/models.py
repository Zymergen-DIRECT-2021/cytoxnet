"""Top level access to toxmodel options from various sources.

Users can access the class within this module to instantiate tox models from
eg. sklearn or deepchem. They can also list available classes or get help.

Example
-------
>>>ToxModel.help()
============================
AVAILABLE TOX MODELS CLASSES
============================
GPR: (sklearn) Gaussian Process Regressor. Accepts vector features.
DTNN: (deepchem) Deep Tensor Neural Network Regressor/Classifier. Accepts
    coulomb matrix features.
...
>>>ToxModel.help('GPR')
GPR (sklearn)
Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
...
>>>mymodel = ToxModel('GPR')
>>>type(mymodel.model)
sklearn.gaussian_process._gpr.GaussianProcessRegressor
"""
# deepchem sklearn model wrapper might be helpful

from typing import Type, Union

import deepchem
import sklearn
import tensorflow.keras

# typing
Model = Type[deepchem.model]
Dataset = Type[deepchem.dataset]

# a codex containing the available models and their information to grab
# dict of `name`: (`short_descr`, `class_string`)
_models = {
    "GPR": (
        "(sklearn) Gaussian Process Regressor. Accepts vector features.",
        "sklearn.gaussian_process.GaussianProcessRegressor",
    ),
    "GPC": (
        "(sklearn) Gaussian Process Classifier. Accepts vector features.",
        "sklearn.gaussian_process.GaussianProcessClassifier",
    )
}


class ToxModel:
    """Highlevel access to available model classes.

    Class instantialization will retrieve the requested model type. The help
    method provides quick access to model docs.

    Parameters
    ----------
        model_name : str
            The name of the model type to instatialize.
        kwargs
            Keyword arguments to pass to the model type for initialization.

    Returns
    -------
        Instance of requested class.
    Example
    -------
    >>>ToxModel.help()
    ============================
    AVAILABLE TOX MODELS CLASSES
    ============================
    GPR: (sklearn) Gaussian Process Regressor. Accepts vector features.
    DTNN: (deepchem) Deep Tensor Neural Network Regressor/Classifier. Accepts
        coulomb matrix features.
    ...
    >>>ToxModel.help('GPR')
    GPR (sklearn)
    Gaussian process regression (GPR).

        The implementation is based on Algorithm 2.1 of Gaussian Processes
        for Machine Learning (GPML) by Rasmussen and Williams.

        In addition to standard scikit-learn estimator API,
        GaussianProcessRegressor:

           * allows prediction without prior fitting (based on the GP prior)
           * provides an additional method sample_y(X), which evaluates samples
             drawn from the GPR (prior or posterior) at given inputs
           * exposes a method log_marginal_likelihood(theta), which can be used
             externally for other ways of selecting hyperparameters, e.g., via
             Markov chain Monte Carlo.

        Read more in the :ref:`User Guide <gaussian_process>`.

        .. versionadded:: 0.18

        Parameters
        ----------
    ...
    >>>mymodel = ToxModel('GPR')
    >>>type(mymodel.model)
    sklearn.gaussian_process._gpr.GaussianProcessRegressor
    """

    models = _models
    """dict: Codex of available models.

    Dictionary of `model_name`: (`model_class`, `short_description`). Models
    are imported and instantialized from this dictionary. The short
    description is used for the help method to give some information.
    """

    def __init__(self,
                model_name: str,
                **kwargs):
        # pseudocode
        # >check model type is available in dict
        # >import model class, use method
        # >if model class is keras type, wrap to DC
        # >same if sklearn
        # >check if "mode" is keyword to class
        # >>if user did not pass "mode", print warning about which mode will
        #       be used eg classification
        # >instantialize model type with keyword
        # >return instance

        # checking and retrieving model class
        if type(model_name) != str:
            raise TypeError(
                'Model name should be type str, not {}'.format(
                    type(model_name)
                )
            )
        else:
            pass
        self._check_model_avail(model_name)
        ModelClass = self._import_model_type(self.models[model_name][1])
        # initialize the model
        model = ModelClass(**kwargs)

        # if the model is already deepchem, check and handle if classify or
        # regress was chosen
        if isinstance(model, deepchem.models.Model):
            if hasattr(model, 'mode') and 'mode' not in kwargs.keys():
                print(
                    'WARNING: `mode` not passed so using the default\
for the task: {}'.format(model.mode)
                )
            self.model = model
        # if the model was sklearn, wrap
        elif isinstance(model, sklearn.BaseEstimator):
            self.model = deepchem.models.SklearnModel(model)
            
        return 

    def _check_model_avail(self, model_name: str):
        """Check if model name is one of the available models.

        Parameters
        ----------
            model_name : str
                The name of the model type to check.
        """
        if model_name not in self.models.keys():
            raise AttributeError(
                "The requested model '{}' is not an available model. Current\
 available models: {}".format(model_name, list(self.models.keys()))
            )
        else:
            return

    def _import_model_type(self, model_type: str):
        """Import model type class from model type module.

        Parameters
        ----------
            model_type : str
                String of model class import path.

        Returns
        ----------
            model_class : object
                The requested model class.
        """
        # pseudocode
        # >importlib all module in package with `list`
        # maybe string.split('.') will do it
        # get package and subpackage names
        components = model_type.split('.')
        # import package
        mod = __import__(components[0])
        # import subpackages
        for i, comp in enumerate(components[1:]):
            if not hasattr(mod, comp):
                raise AttributeError(
                    'The module {} does not contain the attribute {}'.format(
                        components[:i], comp
                    )
                )
            mod = getattr(mod, comp)
        return mod

    def help(model_name: str = None):
        """Get list of available model classes, or help on specific one.

        If no models are specified, prints the list of available models
        with a short description. If an available model name is specified,
        prints the docs for that model class.

        Parameters
        ----------
            model_name : str, default None
                The name of the model type to get help on.
        """
        # pseudocode
        # >if model name is none, print names and short descrs
        # >otherwise print docs for the requested model name
        # the user wants general help on available models
        if model_name == None:
            print('=================')
            print('AVAILABLE MODELS:')
            print('=================')
            for name, (desc, mod) in self.models.items():
                print(name+': ', desc)
        # the user wants help on a specific model type
        elif type(model_name) == str:
            avail_models = list(self.models.keys())
            if model_name not in avail_models:
                raise AttributeError(
                    '{} not an available model from: {}'.format(
                        model_name, avail_models
                    )
                )
            else:
                pass
            # need to import the class first
            ModelClass = self._import_model_type(self.models[model_name][1])
            print(model_name)
            print(ModelClass.__docs__)
        return


def transfer(model: Model,
             dataset: Dataset,
             **kwargs):
    """Transfer a model unto a new task.

    Already learned inner layers of a model are fixed to preserve the learned
    relationship, and outer/pooling layers are appended and trainable. The
    model is then retrained on the new task.

    Parameters
    ----------
        model : :obj:deepchem.model
            The model to conduct transfer learning on.
        dataset : :obj:deepchem.dataset
            The dataset represening the task to transfer onto.

    Returns
    -------
        new_model : :obj:deepchem.model
            The transfered model.
    """
    # pseudo code
    # >check inputs
    # >cut model ties to disk - don't want to overwrite the old model
    # >append/replace output layers? this will be tough and is unclear how to
    #   do this uniformly across model types. Potentially probing of output
    #   layers for types
    # >fix old layers
    # >call fit on new data
    new_model = None
    return new_model


def pretrain(model: Model,
             dataset: Union[Dataset, str],
             **kwargs):
    """Train a model and prepare it to recieve a new task.

    Trains a model on a large/known dataset, then prepares the model to be
    transfered onto a new dataset by fixing inner layers and replacing or
    appending outer layers.

    Parameters
    ----------
        model : :obj:deepchem.model
            The model to pretrain.
        dataset : :obj:deepchem.dataset or str
            The dataset represening the task to transfer onto, or the string
                name of one in the package to use.

    Returns
    -------
        model : :obj:deepchem.model
            The pretrained model.
    """
    # pseudo code
    # >check inputs
    # >call fit on pretraining data
    # >append/replace output layers? this will be tough and is unclear how to
    #   do this uniformly across model types. Potentially probing of output
    #   layers for types
    # >fix old layers
    model = None
    return model
