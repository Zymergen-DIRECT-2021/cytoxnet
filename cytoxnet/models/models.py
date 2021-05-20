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
...
>>>mymodel = ToxModel('GPR', alpha = 1e-4, **kwargs)
>>>type(mymodel.model)
sklearn.gaussian_process._gpr.GaussianProcessRegressor
"""

import importlib
from typing import Type, Union, List

import altair as alt
import deepchem
import numpy
import sklearn

import cytoxnet.models.analyze

# typing
Model = Type[deepchem.models.Model]
Dataset = Type[deepchem.data.Dataset]
Metric = Type[deepchem.metrics.Metric]
Transformer = Type[deepchem.trans.Transformer]
Viz = Type[alt.Chart]

# a codex containing the available models and their information to grab
# dict of `name`: (`short_descr`, `class_string`)
_MODELS = {
    "GPR": (
        "(sklearn) Gaussian Process Regressor. Accepts vector features.",
        "sklearn.gaussian_process.GaussianProcessRegressor",
    ),
    "GPC": (
        "(sklearn) Gaussian Process Classifier. Accepts vector features.",
        "sklearn.gaussian_process.GaussianProcessClassifier",
    ),
    "GraphCNN": (
        "(deepchem) Graph Convolutional Neural Network. Accepts graph\
 features.",
        "deepchem.models.GraphConvModel"
    ),
    "LASSO": (
        "(sklearn) Least Absolute Shrinkage and Selection Operator. Accepts\
 vector features",
        "sklearn.linear_model.Lasso"
    ),
    "RFR": (
        "(sklearn) Random Forest Regressor. Accepts vector features.",
        "sklearn.ensemble.RandomForestRegressor"
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
        transformers : list of :obj:deepchem.transformers.RawTransformer
            Data transformations to apply to output predictions. If the
            training data was transformed/preprocessed, this will allow
            predictions and evaluation to be done in the raw data space.
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

    models = _MODELS
    """dict: Codex of available models.

    Dictionary of `model_name`: (`model_class`, `short_description`). Models
    are imported and instantialized from this dictionary. The short
    description is used for the help method to give some information.
    """

    def __init__(self,
                 model_name: str,
                 tasks: List[str] = None,
                 transformers: List[Transformer] = None,
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
        if not isinstance(model_name, str):
            raise TypeError(
                'Model name should be type str, not {}'.format(
                    type(model_name)
                )
            )
        else:
            pass
        ToxModel._check_model_avail(model_name)
        ModelClass = ToxModel._import_model_type(self.models[model_name][1])
        # save the tasks
        if tasks is None:
            print('WARNING: No tasks passed, assuming one target')
            tasks = ['target', ]
        else:
            assert isinstance(tasks, list),\
                "tasks must be list, not {}".format(type(tasks))
            assert all([isinstance(task, str) for task in tasks]),\
                "tasks should all be string names"
        self.tasks = tasks

        # if the model is already deepchem, check and handle if classify or
        # regress was chosen
        if issubclass(ModelClass, deepchem.models.Model):
            # initialize the model
            model = ModelClass(n_tasks=len(self.tasks), **kwargs)
            if hasattr(model, 'mode') and 'mode' not in kwargs.keys():
                print(
                    'WARNING: `mode` not passed so using the default\
for the task: {}'.format(model.mode)
                )
            self.model = model
        # if the model was sklearn, wrap
        elif issubclass(ModelClass, sklearn.base.BaseEstimator):
            model = ModelClass(**kwargs)
            self.model = deepchem.models.SklearnModel(model)

        # save transformers
        if transformers is not None:
            self.transformers = transformers

        # set top level class methods that do not need to be modified
        self.fit = self.model.fit
        return

    def _check_model_avail(model_name: str):
        """Check if model name is one of the available models.

        Parameters
        ----------
            model_name : str
                The name of the model type to check.
        """
        if model_name not in ToxModel.models.keys():
            raise AttributeError(
                "The requested model '{}' is not an available model. Current\
 available models: {}".format(model_name, list(ToxModel.models.keys()))
            )
        else:
            return

    def _import_model_type(model_type: str):
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
        print(model_type)
        components = model_type.split('.')
        print(components)
        print('looping')
        # import package
        mod = importlib.import_module('.'.join(components[:-1]))
        # import subpackages
        Model = getattr(mod, components[-1])
        return Model

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
        if model_name is None:
            print('=================')
            print('AVAILABLE MODELS:')
            print('=================')
            for name, (desc, mod) in ToxModel.models.items():
                print(name + ': ', desc)
        # the user wants help on a specific model type
        elif isinstance(model_name, str):
            ToxModel._check_model_avail(model_name)
            # need to import the class first
            ModelClass = ToxModel._import_model_type(
                ToxModel.models[model_name][1]
            )
            print('Tox model: ', model_name)
            help(ModelClass)
        return

    def _get_transformers(self, untransform):
        """Retrieve the transformers if untransform specified."""
        if not untransform:
            transformers = []
        else:
            assert hasattr(self, 'transformers'),\
                "untransform was specifed but no transformers saved at the\
 transform attribute."
            transformers = self.transformers
        return transformers

    def predict(self,
                dataset: Dataset,
                untransform: bool = False) -> numpy.ndarray:
        """Makes predictions on dataset.

        Wrapper of deepchem predict method.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.
        untransform: bool
            Untransform predictions with the transformers saved in the
            `transformers` attribute.
        """
        transformers = self._get_transformers(untransform)
        predictions = self.model.predict(dataset, transformers=transformers)
        return predictions

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Union[str, Metric]],
                 untransform: bool = False,
                 per_task_metrics: bool = False,
                 use_sample_weights: bool = False,
                 n_classes: int = None,
                 **kwargs) -> dict:
        """Evaluates the performance of this model on specified dataset.

        Option wrapper of deepchem evaluate method.
        This function uses `Evaluator` under the hood to perform model
        evaluation. As a result, it inherits the same limitations of
        `Evaluator`. Namely, that only regression and classification
        models can be evaluated in this fashion. For generator models, you
        will need to overwrite this method to perform a custom evaluation.

        Keyword arguments specified here will be passed to
        `Evaluator.compute_model_performance`.

        Parameters
        ----------
        dataset: Dataset
          Dataset object.
        metrics: Metric / List[Metric/str]
          The set of metrics provided.
        untransform: bool
            Untransform predictions with the transformers saved in the
            `transformers` attribute.
        per_task_metrics: bool, optional (default False)
          If true, return computed metric for each task on multitask dataset.
        use_sample_weights: bool, optional (default False)
          If set, use per-sample weights `w`.
        n_classes: int, optional (default None)
          If specified, will use `n_classes` as the number of unique classes
          in `self.dataset`. Note that this argument will be ignored for
          regression metrics.

        Returns
        -------
        multitask_scores: dict
          Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
          If `per_task_metrics == True` is passed as a keyword argument,
          then returns a second dictionary of scores for each task
          separately.
        """
        transformers = self._get_transformers(untransform)

        # transform string metrics to class
        metrics_ = []
        for metric in metrics:
            if isinstance(metric, str):
                try:
                    metric_ = getattr(deepchem.metrics, metric)
                except AttributeError:
                    raise ValueError(
                        '{} not a valid metric.'.format(metric)
                    )
            elif callable(metric):
                metric_ = metric
            else:
                raise TypeError(
                    'Cannot use input of type {} as a metric'.format(
                        type(metric)
                    )
                )
            metrics_.append(metric_)

        returns = self.model.evaluate(dataset,
                                      metrics=metrics_,
                                      transformers=transformers,
                                      per_task_metrics=per_task_metrics,
                                      use_sample_weights=use_sample_weights,
                                      n_classes=n_classes)
        return returns

    def visualize(self,
                  viz_name: Union[str, object],
                  dataset: Dataset,
                  **kwargs) -> Viz:
        """Vizualize the model results on a dataset.

        Uses a function accepting ToxModel instance and a dataset as the first
        two positional arguments. Functions from cytoxnet.models.analyze can
        be called by name.

        Parameters
        ----------
            viz_name : str or callable
                The name of a visualization function in the package or a
                function itself to use.
            dataset : deepchem.data.Dataset
                The dataset to use for visualization.
            **kwargs passed to the visualization function
        """
        if not callable(viz_name):
            assert isinstance(viz_name, str),\
                "If not callable, viz_name must be a string name of function."
            try:
                func = getattr(cytoxnet.models.analyze, viz_name)
            except AttributeError:
                raise AttributeError(
                    "{} not a valid function name in cytoxnet.models.analyze"
                    .format(viz_name)
                )
        else:
            func = viz_name

        outs = func(model=self, dataset=dataset, **kwargs)
        return outs

    @property
    def model(self):
        """:obj:deepchem.models.Model : model being used for tox prediction."""
        return self._model

    @model.setter
    def model(self, new_model):
        assert isinstance(new_model, deepchem.models.Model),\
            "Input of type {} is not a deepchem model.".format(type(new_model))
        self._model = new_model
        return

    @property
    def transformers(self):
        """list of :obj:deepchem.transformers.Transformer

        Transformations done to training data to reverse transform outputs.
        """
        return self._transformers

    @transformers.setter
    def transformers(self, new_transformers):
        if isinstance(new_transformers, list):
            pass
        else:
            raise TypeError('transformers must be list')
        for t in new_transformers:
            assert isinstance(t, deepchem.trans.Transformer),\
                "Cannot set transformers, not a deepchem transformer"
        self._transformers = new_transformers
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
