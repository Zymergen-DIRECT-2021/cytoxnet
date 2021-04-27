"""Functions to analyze model performance."""

import pytest
import unittest.mock as mock

import deepchem
import numpy as np

import cytoxnet.models.analyze
import cytoxnet.models.models

def test_pair_predict():
    """Plot of predictions vs true values."""
    # set up a mock model to use
    model = mock.MagicMock(spec=cytoxnet.models.models.ToxModel)
    model.tasks = ['target']
    model.predict.return_value = np.array([1,2,3,4,5]).reshape(-1,1)
    dataset = deepchem.data.NumpyDataset(
        X=np.array(np.random.random((5,2))),
        y=np.array([1,3,3,4,5]).reshape(-1,1)
    )
    transformer = mock.MagicMock()
    transformer.untransform.return_value = np.array(
        [1,2,3,4,5]
    ).reshape(-1,1)
    with mock.patch('cytoxnet.models.analyze.alt') as mocked_altair:
        # no transform should work
        cytoxnet.models.analyze.pair_predict(
            model, dataset, untransform=False)
        # assign transformer and use
        model.transformers = [transformer]
        cytoxnet.models.analyze.pair_predict(
            model, dataset, untransform=True)
        transformer.untransform.assert_called_with(
            dataset.y
        )
        # and task handling
        model.tasks = ['t1', 't2']
        with pytest.raises(AssertionError):
            cytoxnet.models.analyze.pair_predict(
                model, dataset)
        # specify it this time
        model.predict.return_value = np.array(
            [[1,2,3,4,5], [1,2,3,4,5]]
        ).T
        dataset = deepchem.data.NumpyDataset(
            X=np.array(np.random.random((5,2))),
            y=np.array(
                [[1,2,3,4,5], [1,2,3,4,5]]
            ).T
        )
        chart = cytoxnet.models.analyze.pair_predict(
            model, dataset, task='t1', untransform=False)
        assert chart is not None,\
            "Nothing was returned from the visualization."
    return