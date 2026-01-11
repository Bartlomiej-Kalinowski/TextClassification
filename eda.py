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
        df = self.dataset.drop_duplicates()
        df.count()

    def handle_null_data(self):
        print((self.dataset.isnull().sum()))


def main():
    ld = Loader(
        'kaggle',
        "ashfakyeafi/spam-email-classification",  # dataset 1 - spam ham emails classification
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",  # dataset 2 - movie reviews sentiment classification
        "aadyasingh55/fake-news-classification"  # fake_news classification
    )
    ld.load()

    print("\n\nDataset emails spam ham:\n\n")
    ds_emails = Eda(os.path.join(ld.download_paths[0], 'email.csv'))
    ds_emails.display_info_about_dataset()
    ds_emails.dr_duplicates()
    ds_emails.handle_null_data()

    print("\n\nDataset movie reviews sentiment:\n\n")
    ds_sentiment = Eda(os.path.join(ld.download_paths[1], 'IMDB Dataset.csv'))
    ds_sentiment.display_info_about_dataset()
    ds_sentiment.dr_duplicates()
    ds_sentiment.handle_null_data()

    print("\n\nDataset fake news - train:\n\n")
    ds_fake_news = Eda(os.path.join(ld.download_paths[2], 'train (2).csv'), sep = ';')
    ds_fake_news.display_info_about_dataset()
    ds_fake_news.dr_duplicates()
    ds_fake_news.handle_null_data()
    print("\n\nDataset fake news - test:\n\n")
    ds_fake_news_test = Eda(os.path.join(ld.download_paths[2], 'test (1).csv'), sep=';')
    ds_fake_news_test.display_info_about_dataset()
    ds_fake_news_test.dr_duplicates()
    ds_fake_news_test.handle_null_data()


if __name__ == '__main__':
    main()

