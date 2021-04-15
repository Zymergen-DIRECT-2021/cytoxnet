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
>>>type(mymodel)
sklearn.gaussian_process._gpr.GaussianProcessRegressor
"""
# deepchem sklearn model wrapper might be helpful

import importlib

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
    >>>type(mymodel)
    sklearn.gaussian_process._gpr.GaussianProcessRegressor
    """

    models = _models
    """dict: Codex of available models.
    
    Dictionary of `model_name`: (`model_class`, `short_description`). Models
    are imported and instantialized from this dictionary. The short
    description is used for the help method to give some information.
    """

    def __new__(self,
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
        return model
    
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
        """
        # pseudocode
        # >importlib all module in package with `list`
        # maybe string.split('.') will do it
        return

    def help(self, model_name: str = None):
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
        return