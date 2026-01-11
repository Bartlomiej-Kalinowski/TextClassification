from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch_directml

class TfidfPipeline:
    def __init__(self, documents, categories, stop_words_language = 'english'):
        self.language = stop_words_language
        self.vectorizer = TfidfVectorizer(stop_words=self.language)
        self.docs = documents
        self.le = LabelEncoder()
        self.cat = self.le.fit_transform(categories)

    def vectorize(self, transform = True):
        if transform:
            return self.vectorizer.fit_transform(self.docs), self.cat
        else:
            return self.vectorizer.transform(self.docs), self.cat

class EmbeddingPipeline:
    def __init__(self, documents, categories, stop_words_language = 'english'):
        self.language = stop_words_language
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = documents.tolist()
        self.le = LabelEncoder()
        self.cat = self.le.fit_transform(categories)
        # LLM
        # Sprawdzamy czy DirectML jest dostępny
        try:
            self.device = torch_directml.device()
            print(f"Używam GPU AMD (DirectML): {torch_directml.device_name(0)}")
        except:
            self.device = "cpu"
            print("GPU AMD nieodnalezione, wracam do CPU.")

        self.model.to(self.device)

    def vectorize(self):
        print(1)
        embeddings = self.model.encode(self.docs, device=self.device)
        print(2)
        return embeddings


