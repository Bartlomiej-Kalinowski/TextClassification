from dataLoader import Loader
import pandas as pd
import os

class Eda:
    def __init__(self, path_to_downoladed_train_dataset, sep = ','):
        self.dataset = pd.read_csv(path_to_downoladed_train_dataset, encoding='utf-8', sep = sep)

    def display_info_about_dataset(self):
        print(self.dataset.shape)
        print(self.dataset.dtypes)
        print(self.dataset.head(3))
        print(self.dataset.tail(3))
        print(self.dataset.count())

    def dr_duplicates(self):
        duplicate_rows = self.dataset[self.dataset.duplicated()]
        print(self.dataset[self.dataset.duplicated()].head(5))
        print("number of duplicate rows: ", duplicate_rows.shape[0])
        self.dataset = self.dataset.drop_duplicates()
        self.dataset.count()

    def handle_null_data(self):
        print((self.dataset.isnull().sum()))
