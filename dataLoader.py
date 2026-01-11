import kagglehub
import os
import shutil

class Loader:
    def __init__(self, src = 'kaggle', *args):
        self.source = src
        self.paths = []
        for path in args:
            self.paths.append(path)
        self.download_paths = []
        self.train_dir = r"C:\Users\Kalin\PycharmProjects\pythonTextClassifier\dataset"

    def load_from_kaggle(self):
        for path in self.paths:
            self.download_paths.append(kagglehub.dataset_download(path))
        i = 1
        for dw_path in self.download_paths:
            new_path = os.path.join(self.train_dir, f"dataset{i}")
            # LLM
            shutil.copytree(dw_path, new_path, dirs_exist_ok=True)
            self.download_paths[i - 1] = new_path
            # LLM - end
            i += 1
        print("Path to downloaded datasets: ", self.download_paths)


    def load(self):
        if self.source.lower() == 'kaggle':
            self.load_from_kaggle()


def main():
    ld = Loader(
        'kaggle',
        "ashfakyeafi/spam-email-classification", # dataset 1 - spam ham emails classification
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",# dataset 2 - movie reviews sentiment classification
        "aadyasingh55/fake-news-classification"# fake_news classification
    )
    ld.load()

if __name__ == '__main__':
    main()