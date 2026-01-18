from dataLoader import Loader


def test_load_from_kaggle(mocker):
    mock_kaggle = mocker.patch('kagglehub.dataset_download', return_value='/tmp/fake_path')
    mock_copy = mocker.patch('dataLoader.shutil.copytree')
    ld = Loader('kaggle', 'test/dataset-name')
    ld.load()
    mock_kaggle.assert_called_once_with('test/dataset-name')
    mock_copy.assert_called()