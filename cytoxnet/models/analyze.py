"""Visual analysis of model performance.

A compilation of visualization tools to help the analysis of model performance.

Examples
--------
>>>my_model = ToxModel('GPR')
>>>my_model.fit(train_set)
>>>visualization = pair_predict(my_model, test_set)
"""
from typing import Type

import altair as alt
import deepchem.data
import numpy as np
import pandas
import deepchem.models
import sklearn.metrics

Dataset = Type[deepchem.data.Dataset]
Viz = Type[alt.Chart]


def pair_predict(model: object,
                 dataset: Dataset,
                 task: str = None,
                 untransform: bool = True,
                 return_df: bool = False) -> Viz:
    """Plot pairwise regression predictions.

    Plots the model's predicted values against its true values, as well as
    the fit line of predictions against the x = y perfect model.

    Parameters
    ----------
        model : object with predict method
            The tox model to evaluate.
        dataset : :obj:deepchem.data.Dataset
            The testing dataset to be used for evaluation
        task : str, default None
            Which task to use for plotting - ignored for single task models.
        untransform : bool, default True
            Whether or not to plot untransformed target values.
        return_df : bool, default False
            Whether to return a dataframe of predictions and true values.

    Returns
    -------
        chart : :obj:altair.Chart
            The visualization of predictions.
    """
    # check inputs
    assert hasattr(model, 'predict'),\
        "`model` does not have a `predict` method"
    assert isinstance(dataset, deepchem.data.Dataset),\
        "`dataset` is not a dataset object."
    if len(model.tasks) > 1:
        assert task in model.tasks,\
            "The problem is multitasked, but the passed task does not\
 correspond to one of them"
    else:
        task = model.tasks[0]
    # make predictions - a numpy array
    predictions = np.vstack(model.predict(dataset, untransform=untransform))
    print(predictions.shape)
    if untransform:
        assert hasattr(model, 'transformers'),\
            "untransform specified but the model has no transformers"
        X = dataset.X
        y = dataset.y
        for trans in model.transformers:
            if trans.transform_y:
                y = trans.untransform(y)
            if trans.transform_X:
                X = trans.untransform(X)
    else:
        y = dataset.y
    y = np.vstack(y)
    assert predictions.shape == y.shape,\
        "predictions and true values should have the same shape."
    assert y.shape[-1] == len(model.tasks),\
        "Number targets predicted does not match number of tasks."

    # handle tasks
    df = pandas.DataFrame()
    for i, task_ in enumerate(model.tasks):
        df[task_ + ': true'] = y[:, i]
        df[task_ + ': predicted'] = predictions[:, i]

    # x and y titles
    xl = task + ': true'
    yl = task + ': predicted'

    # make the plot
    chart = alt.Chart(df).mark_point().encode(
        alt.X(xl + ':Q', scale=alt.Scale(zero=False)),
        alt.Y(yl + ':Q', scale=alt.Scale(zero=False))
    )
    chart = chart + chart.transform_regression(
        xl, yl
    ).mark_line()

    # add a x=y line
    xmin, xmax = df[xl].min(), df[xl].max()
    ymin, ymax = df[yl].min(), df[yl].max()
    if ymax - ymin > xmax - xmin:
        ran = ymin, ymax
    else:
        ran = xmin, xmax

    linedf = pandas.DataFrame({xl: [*ran], yl: [*ran]})

    line = alt.Chart(linedf).mark_line(color='black').encode(
        alt.X(xl + ':Q'),
        alt.Y(yl + ':Q')
    )
    chart = chart + line
    if return_df:
        return chart, df
    else:
        return chart

def roc(model: object,
        dataset: Dataset,
        task: str = None,
        return_df: bool = False) -> Viz:
    
    preds_raw = model.predict(dataset)
    if len(model.tasks) == 1:
        task = model.tasks[0]
    # handle probabilities returns
    if len(preds_raw.shape) > 2:
        assert task is not None, 'Multitask model but no tasks passed'
        preds = np.argmax(preds_raw, axis=2)
    elif len(preds_raw.shape) == 2:
        preds = np.argmax(preds_raw, axis=1)
    else:
        preds = preds_raw
    preds = np.atleast_2d(preds)
    
    # handle tasks
    df = pandas.DataFrame()
    for i, task_ in enumerate(model.tasks):
        df[task_ + ': true'] = dataset.y[:, i]
        if isinstance(model.model, deepchem.models.SklearnModel):
            df[task_ + ': predicted'] = preds[i, :]
        else:
            df[task_ + ': predicted'] = preds[:, i]
    
    taskind = np.where(model.tasks == task)[0]
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        df[task + ': true'], df[task + ': predicted']
    )    
    roc_df = pandas.DataFrame()
    roc_df['fpr'] = fpr
    roc_df['tpr'] = tpr
    roc_df['thresholds'] = thresholds
        
        
    chart = alt.Chart(roc_df).mark_area(
        fillOpacity = 0.5, fill = 'red').encode(
            alt.X('fpr', title="false positive rate"),
            alt.Y('tpr', title="true positive rate")
        )
    line =  alt.Chart(pandas.DataFrame({'line': [0,1]})).mark_line(
        strokeDash=[20,5], color = 'black'
    ).encode(
        alt.X('line', title="false positive rate",
              scale = alt.Scale(domain=[0.0, 1.0])),
        alt.Y('line', title="true positive rate",
              scale = alt.Scale(domain=[0.0, 1.0])))
    chart = line + chart
    chart.properties(
        title='ROC curve')
    if return_df:
        return chart, df
    else:
        return chart
