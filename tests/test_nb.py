import numpy as np
from nb import ClassifierNB
import pytest

#LLM
def test_split_creates_train_and_test_sets(mocker):
    mocker.patch('nb.ClassifierNB.__init__', return_value=None)

    # 2. Tworzymy "pusty" obiekt
    clf = ClassifierNB(None, None)

    X = np.random.rand(10, 5)
    y = np.array([0, 1] * 5)
    clf.data_to_classify = (X, y)

    # ACT
    clf.split()

    # ASSERT
    assert clf.x_train is not None
    assert clf.x_test is not None
    assert clf.y_train is not None
    assert clf.y_test is not None

    assert len(clf.x_train) + len(clf.x_test) == 10
    assert len(clf.y_train) + len(clf.y_test) == 10
#LLm-end

def test_fit_logic_split_trigger(mocker):
    mocker.patch('nb.ClassifierNB.__init__', return_value=None)
    clf = ClassifierNB(None, None)
    clf.classifier = 'MNB'
    clf.gnb = mocker.Mock()
    clf.y_train = [1]

    #1
    clf.x_train =None
    mock_split = mocker.patch.object(clf, 'split')
    clf.fit()
    mock_split.assert_called_once()
    #2
    clf.x_train = [1, 2, 5]
    mock_split.reset_mock()
    clf.fit()
    mock_split.assert_not_called()
    x = np.random.rand(10, 5)
    y = np.array([0, 1] * 5)
    clf.data_to_classify = (x, y)

#LLM------------------------------------------------------------------------------------
@pytest.mark.parametrize("classifier, has_toarray, expected_conversion", [
    ('GNB', True, True),
    ('GNB', False, False),
    ('MNB', True, False),
])
def test_predict_conversion_logic(mocker, classifier, has_toarray, expected_conversion):
    # Setup
    mocker.patch('nb.ClassifierNB.__init__', return_value=None)
    model = ClassifierNB(None, None)  # Muszą być 2 arg
    model.classifier = classifier
    model.gnb = mocker.Mock()
    model.y_test = [0, 1]

    # Mockujemy LabelEncoder w pipeline
    model.pipeline = mocker.Mock()
    model.pipeline.le.inverse_transform.return_value = ["cat1", "cat2"]

    if has_toarray:
        mock_x = mocker.Mock()
        mock_x.toarray.return_value = np.array([[1], [2]])
        model.x_test = mock_x
    else:
        model.x_test = mocker.Mock(spec=[])  # Brak toarray

    # Act
    model.predict()

    # Assert
    if expected_conversion:
        model.x_test.toarray.assert_called_once()
    elif hasattr(model.x_test, 'toarray'):
        model.x_test.toarray.assert_not_called()

def test_visualize_execution(mocker):
    """
    Testujemy czy visualize przechodzi przez całą ścieżkę bez błędów
    i wywołuje kluczowe funkcje rysujące.
    """
    # Setup
    mocker.patch('nb.ClassifierNB.__init__', return_value=None)
    model = ClassifierNB(None, None)
    model.classifier = "GNB"

    # Przygotowujemy sztuczne dane (2 próbki, 3 cechy)
    model.x_train = np.random.rand(5, 3)
    model.x_test = np.random.rand(2, 3)
    model.y_train = np.array([0, 1, 0, 1, 0])
    model.y_test = np.array([1, 0])

    # Mockujemy matplotlib, żeby nie otwierał okienek podczas testów
    mock_plt = mocker.patch('matplotlib.pyplot.show')

    # Mockujemy TSNE i funkcje plotujące z bibliotek zewnętrznych
    mocker.patch('nb.TSNE.fit_transform', return_value=np.random.rand(7, 2))
    mock_regions = mocker.patch('nb.plot_decision_regions')
    mock_learning = mocker.patch('nb.plot_learning_curves')

    # Act
    model.visualize()

    # Assert
    assert mock_plt.call_count == 2  # Dwa wykresy powinny zostać pokazane
    mock_regions.assert_called_once()
    mock_learning.assert_called_once()
#LLm-end -------------------------------------------------------------------------



    


