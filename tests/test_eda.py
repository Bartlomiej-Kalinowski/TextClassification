from eda import Eda
import pandas as pd

#LLM
def test_dr_duplicates_actually_removes_rows(mocker):
    # 1. Przygotowanie danych z duplikatami
    data = {'col1': [1, 1, 2], 'col2': [3, 3, 4]}
    df = pd.DataFrame(data)

    # 2. Mockujemy pd.read_csv, żeby nie czytał z dysku, tylko zwrócił nasz df
    mocker.patch('pandas.read_csv', return_value=df)

    # 3. Uruchamiamy kod
    eda = Eda("fake_path.csv")
    initial_shape = eda.dataset.shape[0]  # 3 wiersze
    eda.dr_duplicates()
    final_shape = eda.dataset.shape[0]  # powinny być 2 wiersze

    # 4. Sprawdzamy czy duplikaty zniknęły
    assert initial_shape == 3
    assert final_shape == 2