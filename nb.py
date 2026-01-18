from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from mlxtend.plotting import plot_learning_curves, plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from vectorizer import TfidfPipeline, EmbeddingPipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sys
import numpy as np


class ClassifierNB:
    def __init__(self, language, classifier, documents=None, categories=None, x_train=None, x_test=None, y_train=None,
                 y_test=None):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None  # POPRAWKA: Przechowywanie wyników do raportu
        self.classifier = classifier

        if (documents is None or categories is None) and (
                x_train is None or x_test is None or y_train is None or y_test is None):
            print("Błąd: Brak danych do klasyfikacji")
            sys.exit(0)

        if self.classifier == 'MNB':
            self.gnb = MultinomialNB()
            if documents is not None and categories is not None:
                self.pipeline = TfidfPipeline(documents, categories, stop_words_language=language)
                # POPRAWKA: TfidfPipeline.vectorize zwraca (X, y)
                self.data_to_classify = self.pipeline.vectorize(transform=True)
            else:
                self.pipeline = TfidfPipeline(x_train, y_train, stop_words_language=language)
                self.x_train, self.y_train = self.pipeline.vectorize(transform=True)
                self.x_test = self.pipeline.vectorizer.transform(x_test)
                self.y_test = self.pipeline.le.transform(y_test)
        else:
            self.gnb = GaussianNB()
            if documents is not None and categories is not None:
                self.pipeline = EmbeddingPipeline(documents, categories, stop_words_language=language)
                # self.pipeline.vectorize() zwraca (embeddings, categories)
                self.data_to_classify = self.pipeline.vectorize()
            else:
                self.pipeline = EmbeddingPipeline(x_train, y_train, language)
                self.x_train, self.y_train = self.pipeline.vectorize()
                # Przy testowym nie robimy fit_transform, tylko transform (vectorize w EmbeddingPipeline używa modelu)
                pipeline_test = EmbeddingPipeline(x_test, y_test, language)
                self.x_test, _ = pipeline_test.vectorize()
                self.y_test = self.pipeline.le.transform(y_test)

    def split(self):
        # POPRAWKA: Zabezpieczenie przed brakiem danych
        if hasattr(self, 'data_to_classify'):
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.data_to_classify[0], self.data_to_classify[1], test_size=0.2, random_state=42)

    def fit(self):
        print(f'Trenowanie modelu {self.classifier}...')
        if self.x_train is None:
            self.split()

        x_train_fit = self.x_train
        if self.classifier == 'GNB' and hasattr(x_train_fit, 'toarray'):
            x_train_fit = x_train_fit.toarray()

        self.gnb.fit(x_train_fit, self.y_train)

    def predict(self):
        x_test_pred = self.x_test
        if self.classifier == 'GNB' and hasattr(x_test_pred, 'toarray'):
            x_test_pred = x_test_pred.toarray()

        self.y_pred = self.gnb.predict(x_test_pred)

        # Dekodowanie etykiet dla czytelności raportu
        y_pred_labels = self.pipeline.le.inverse_transform(self.y_pred)
        y_test_labels = self.pipeline.le.inverse_transform(self.y_test)

        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        return self.y_pred

    def visualize(self):
        print("Przygotowywanie wizualizacji...")
        # POPRAWKA: Jeśli nie ma data_to_classify (bo dane były pre-split), budujemy je z train+test
        if not hasattr(self, 'data_to_classify'):
            import scipy.sparse as sp
            if sp.issparse(self.x_train):
                X_all = sp.vstack([self.x_train, self.x_test])
            else:
                X_all = np.vstack([self.x_train, self.x_test])
            y_all = np.concatenate([self.y_train, self.y_test])
        else:
            X_all, y_all = self.data_to_classify

        # TSNE potrzebuje gęstej macierzy dla niektórych wersji lub parametrów
        X_dense = X_all.toarray() if hasattr(X_all, 'toarray') else X_all

        # Redukcja wymiarów
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X_dense)

        # Do wizualizacji granic decyzyjnych w 2D zawsze używamy GaussianNB na zredukowanych danych
        gnb_2d = GaussianNB()
        gnb_2d.fit(X_2d, y_all)

        # 1. Plot Decision Regions
        plt.figure(figsize=(10, 6))
        plot_decision_regions(X=X_2d, y=y_all.astype(int), clf=gnb_2d, legend=2)
        plt.title(f"Regiony decyzyjne (TSNE + {self.classifier})")
        plt.show()

        # 2. Plot Learning Curves
        # Musimy użyć gęstych danych 2D
        x_train_2d, x_test_2d, y_train_2d, y_test_2d = train_test_split(
            X_2d, y_all, test_size=0.3, random_state=42)

        plt.figure(figsize=(10, 6))
        plot_learning_curves(x_train_2d, y_train_2d, x_test_2d, y_test_2d, gnb_2d,
                             print_model=False, style='ggplot')
        plt.title("Krzywa uczenia (Learning Curve)")
        plt.show()