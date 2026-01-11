#LLM-zwiazane z GUI, reszta:wlasna
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QTextEdit, QLabel, QFrame,
                             QMessageBox, QHBoxLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Importy Twoich modułów lokalnych
from dataLoader import Loader
from eda import Eda
from nb import ClassifierNB


class ModernGUI(QMainWindow):
    def __init__(self, trained_model):
        super().__init__()
        # Przypisujemy wytrenowany model do zmiennej klasy
        self.model = trained_model

        self.setWindowTitle("Email Spam Detector - Modern Naive Bayes")
        self.setMinimumSize(700, 600)
        self.setStyleSheet("background-color: #f0f2f5;")
        self.init_ui()

    def init_ui(self):
        """Inicjalizacja nowoczesnego interfejsu użytkownika."""
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Sekcja Nagłówka
        header = QLabel("Wykrywanie Spamu (Email)")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #1c1e21;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        info = QLabel("Model: Multinomial Naive Bayes | Dane: Spam Email Dataset")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #606770; font-size: 14px;")
        layout.addWidget(info)

        # Pole wprowadzania tekstu
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Wklej tutaj treść wiadomości e-mail do analizy...")
        self.text_input.setStyleSheet("""
            QTextEdit { 
                background: white; 
                border-radius: 12px; 
                border: 2px solid #ddd; 
                padding: 20px; 
                font-size: 15px;
                color: #1c1e21;
            }
            QTextEdit:focus { border: 2px solid #0866ff; }
        """)
        layout.addWidget(self.text_input)

        # Układ przycisków
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_run = QPushButton("ANALIZUJ TREŚĆ")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton { 
                background-color: #0866ff; color: white; font-weight: bold; 
                padding: 15px; border-radius: 10px; font-size: 14px;
            }
            QPushButton:hover { background-color: #0055d4; }
        """)
        self.btn_run.clicked.connect(self.classify_single)

        self.btn_viz = QPushButton("POKAŻ WYKRESY I RAPORT")
        self.btn_viz.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_viz.setStyleSheet("""
            QPushButton { 
                background-color: #42b72a; color: white; font-weight: bold; 
                padding: 15px; border-radius: 10px; font-size: 14px;
            }
            QPushButton:hover { background-color: #36a420; }
        """)
        self.btn_viz.clicked.connect(self.show_visualize)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_viz)
        layout.addLayout(btn_layout)

        # Sekcja Wyniku (Karta)
        self.result_card = QFrame()
        self.result_card.setStyleSheet("""
            QFrame { 
                background: white; 
                border-radius: 15px; 
                border: 1px solid #ddd; 
            }
        """)
        res_layout = QVBoxLayout()
        self.result_label = QLabel("Gotowy do analizy")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Medium))
        self.result_label.setStyleSheet("border: none; color: #606770;")
        res_layout.addWidget(self.result_label)
        self.result_card.setLayout(res_layout)
        layout.addWidget(self.result_card)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def classify_single(self):
        """Pobiera tekst i wykonuje predykcję na podstawie wytrenowanego modelu."""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Błąd", "Wprowadź tekst wiadomości!")
            return

        try:
            # 1. Transformacja tekstu przez wektoryzator TF-IDF modelu
            vec = self.model.pipeline.vectorizer.transform([text])

            # 2. Predykcja
            pred_id = self.model.gnb.predict(vec)[0]

            # 3. Dekodowanie etykiety (Spam/Ham)
            label = self.model.pipeline.le.inverse_transform([pred_id])[0]

            # 4. Aktualizacja interfejsu
            if str(label).lower() == "spam":
                color = "#d93025"  # Czerwony
                status = "⚠️ TO JEST SPAM"
            else:
                color = "#188038"  # Zielony
                status = "✅ WIADOMOŚĆ BEZPIECZNA (HAM)"

            self.result_label.setText(status)
            self.result_label.setStyleSheet(f"color: {color}; font-weight: bold; border: none;")

        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Błąd podczas klasyfikacji: {str(e)}")

    def show_visualize(self):
        """Uruchamia okna wizualizacji z pliku nb.py."""
        try:
            self.model.visualize()
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie można wygenerować wykresów: {str(e)}")


def main():
    # 1. Pobieranie danych (Kaggle)
    ld = Loader('kaggle', "ashfakyeafi/spam-email-classification")
    ld.load()

    # 2. Czyszczenie danych (EDA)
    print("\n>>> Przygotowywanie danych...")
    ds_emails = Eda(os.path.join(ld.download_paths[0], 'email.csv'))
    ds_emails.handle_null_data()
    ds_emails.dr_duplicates()

    # Wyświetlamy info o czystych danych w konsoli
    ds_emails.display_info_about_dataset()

    # 3. Trenowanie modelu (Multinomial NB - szybki i skuteczny dla TF-IDF)
    print("\n>>> Trenowanie modelu Naive Bayes... Proszę czekać.")
    spam_model = ClassifierNB('english', 'MNB',
                              ds_emails.dataset['Message'],
                              ds_emails.dataset['Category'])

    # Wywołujemy fit() raz - tutaj model tworzy wektory i uczy się macierzy pomyłek
    spam_model.fit()

    # Tworzymy predykcję testową dla potrzeb wizualizacji (Classification Report)
    spam_model.predict()

    print(">>> Model gotowy. Uruchamiam GUI.")

    # 4. Uruchomienie aplikacji PyQt6
    app = QApplication(sys.argv)

    # Przekazujemy wytrenowany obiekt 'spam_model' do klasy GUI
    gui = ModernGUI(spam_model)
    gui.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

