"""Test ToxModel base class."""
import importlib
import pytest
import unittest.mock as mock

import deepchem
import sklearn.gaussian_process

import cytoxnet.models.models


def test__MODELS():
    "Check whether the format of the models dict is valid"
    subject = cytoxnet.models.models._MODELS
    assert isinstance(subject, dict),\
        "_MODELS variable should be a dict"
    for key, value in subject.items():
        # test that they key is a string - this is really the only test we can
        # do for the key as it is a developer assigned name
        assert isinstance(key, str),\
            "keys in _MODELS should be string names that will be accessed in\
 ToxModels"
        # Value should be a tuple len 2 as set up
        assert isinstance(value, tuple) and len(value) == 2,\
            "values in _MODELS should be a tuple of description, import path"
        descr, impath = value
        # check that the descr is a string to print - best test we can do
        assert isinstance(descr, str),\
            "First item in _MODELS values should be a short string description"
        # import path should be a string also
        # we can also test that it is a real object
        assert isinstance(impath, str),\
            "Second item in _MODELS values should be string import path"
        assert importlib.util.find_spec(
            '.'.join(impath.split('.')[:-1])
        ) is not None,\
            "Could not find the package path in the env at {}".format(impath)
    return


class TestToxModel:

    @mock.patch.dict(cytoxnet.models.models.ToxModel.models,
                     {'bar': ('descr', 'impath')},
                     clear=True)
    def test__check_model_avail(self):
        "Proper access of models attribute."
        cytoxnet.models.models.ToxModel._check_model_avail('bar')  # pass
        # fail
        with pytest.raises(AttributeError):
            cytoxnet.models.models.ToxModel._check_model_avail('foo')
        return

    @mock.patch('deepchem.models.WeaveModel')
    def test__import_model_type(self, mocked_class):
        "Importing the desired model classes."
        ModelClass = cytoxnet.models.models.ToxModel._import_model_type(
            'deepchem.models.WeaveModel'
        )
        assert ModelClass is mocked_class
        return

    def test___init__(self):
        """Ability to recognize models and pass kwargs them."""
        assert cytoxnet.models.models.ToxModel.models is \
            cytoxnet.models.models._MODELS,\
            "ToxModel does not have the models class attribute."
        subject = cytoxnet.models.models.ToxModel('GPR')
        assert isinstance(subject.model, deepchem.models.SklearnModel),\
            "sklearn was not wrapped"
        assert len(subject.tasks) == 1,\
            "tasks not specified so should be length one."
        subject = cytoxnet.models.models.ToxModel(
            'GraphCNN', tasks=['y1', 'y2'])
        assert isinstance(subject.model, deepchem.models.GraphConvModel),\
            "Did not get graph model"
        assert subject.model.n_tasks == 2,\
            'Two tasks specified so should be len 2.'
        # try with a transformer
        # failure wrong type
        with pytest.raises(TypeError):
            subject = cytoxnet.models.models.ToxModel('GPR',
                                                      transformers='string')
        subject = cytoxnet.models.models.ToxModel(
            'GPR', transformers=[
                deepchem.trans.NormalizationTransformer(
                    transform_X=True)])
        assert hasattr(subject, 'transformers')
        return

    def test_help(self):
        """Getting help on models or all models."""
        # specific model
        with mock.patch('cytoxnet.models.models.help') as mocked_help:
            cytoxnet.models.models.ToxModel.help('GPR')
            mocked_help.assert_called_with(
                sklearn.gaussian_process.GaussianProcessRegressor
            )
        # all models
        with mock.patch('cytoxnet.models.models.print') as mocked_print:
            cytoxnet.models.models.ToxModel.help()
            call_list = []
            for name, (desc, mod) in cytoxnet.models.models._MODELS.items():
                call_list.append(mock.call(name + ': ', desc))
            mocked_print.assert_has_calls(
                call_list,
                any_order=True
            )
        return

    def test_fit(self):
        """Fully inherited method, just test that it is present."""
        subject = cytoxnet.models.models.ToxModel('GPR')
        assert hasattr(subject, 'fit'),\
            "sklearn model did not inherit a fit method."
        subject = cytoxnet.models.models.ToxModel('GraphCNN')
        assert hasattr(subject, 'fit'),\
            "deepchem model did not inherit a fit method."
        return

    def test__get_transformers(self):
        """Handling whether to untransform or not."""
        subject = cytoxnet.models.models.ToxModel('GPR')
        # no transform, should return empty list
        assert subject._get_transformers(False) == [],\
            "Empty list should be returned for no untransform"
        # now ask for it but none specified
        with pytest.raises(AssertionError):
            subject._get_transformers(True)
        # patch transformers and retrieve them
        subject._transformers = ['t1', 't2']
        assert subject._get_transformers(True) == ['t1', 't2'],\
            "Asked for untransform but did not return the transformers"
        return

    def test_predict(self):
        """Transform input handling and sub calls."""
        subject = cytoxnet.models.models.ToxModel('GPR')
        # patch over the actual predicter to test calls
        mocked_predictor = mock.MagicMock()
        subject._model = mocked_predictor

        # mock transformers to actually test
        subject._get_transformers = mock.MagicMock(return_value=['t1', 't2'])
        subject.predict('some_data')
        subject._get_transformers.assert_called()
        mocked_predictor.predict.assert_called_with(
            'some_data', transformers=['t1', 't2']
        )
        return

    def test_evaluate(self):
        """Can we get the correct metrics"""
        subject = cytoxnet.models.models.ToxModel('GPR')
        # patch over the actual predicter to test calls
        mocked_predictor = mock.MagicMock()
        subject._model = mocked_predictor
        subject._get_transformers = mock.MagicMock(return_value=['t1', 't2'])
        # patch and use some valid metric names
        with mock.patch('deepchem.metrics.matthews_corrcoef') as mocked_mc:
            metrics = ['matthews_corrcoef']
            subject.evaluate('some_data',
                             untransform=False,
                             metrics=metrics,
                             per_task_metrics=False,
                             use_sample_weights=False,
                             n_classes=None)
            subject.model.evaluate.assert_called_with(
                'some_data',
                transformers=[
                    't1',
                    't2'],
                metrics=[mocked_mc],
                per_task_metrics=False,
                use_sample_weights=False,
                n_classes=None)
            subject._get_transformers.assert_called_with(False)
        # fail bad metric
        with pytest.raises(ValueError):
            subject.evaluate('some_data',
                             metrics=['bad_metric'])
        # pass direct callable

        def func():
            return None
        subject.evaluate('some_data',
                         metrics=[func])
        subject.model.evaluate.assert_called_with('some_data',
                                                  transformers=['t1', 't2'],
                                                  metrics=[func],
                                                  per_task_metrics=False,
                                                  use_sample_weights=False,
                                                  n_classes=2)
        return

    def test_visualize(self):
        """All this does is grab and call function we write elsewhere."""
        # first try with callable
        subject = cytoxnet.models.models.ToxModel('GPR')
        func = mock.MagicMock()
        subject.visualize(func, 'some_data', kword=5)
        func.assert_called_with(dataset='some_data', model=subject, kword=5)
        # try with callable in analyze module by name
        with mock.patch(
            'cytoxnet.models.analyze.a_plotter',
            return_value='plot',
            create=True
        ) as mocked_plotter:
            plot = subject.visualize('a_plotter', 'some_data', kword=5)
            mocked_plotter.assert_called_with(
                model=subject, dataset='some_data', kword=5
            )
            assert plot == 'plot'
        return
