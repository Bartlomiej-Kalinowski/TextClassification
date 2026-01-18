import kagglehub
import os
import shutil
from pathlib import Path

class Loader:
    def __init__(self, src = 'kaggle', *args):
        self.source = src
        self.paths = []
        for path in args:
            self.paths.append(path)
        self.download_paths = []
        BASE_DIR = Path(__file__).resolve().parent

        # Teraz budujesz ścieżkę do datasetu bez względu na to, gdzie jest projekt
        self.train_dir = BASE_DIR / "dataset"

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
        platform = self.source.lower()
        if platform == 'kaggle':
            self.load_from_kaggle()
        else:
            raise ValueError(f"Unsupported platform: {self.source}")