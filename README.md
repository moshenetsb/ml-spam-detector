# Detektor Spamu (ML Spam Detector)

> Projekt ma na celu stworzenie modelu uczenia maszynowego do klasyfikacji wiadomości jako spam lub nie spam

## Zbiory danych

Do przeprowadzenia analizy oraz trenowania modeli w ramach projektu eksperymentalnego wybraliśmy **SMS Spam Collection (UCI)**. Zbiór liczy około 5,5 tys. wiadomości SMS (oznaczonych jako spam lub ham).

## Jak uruchomić projekt

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/moshenetsb/ml-spam-detector.git
   cd ml-spam-detector
   ```

2. **Utwórz i aktywuj wirtualne środowisko:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Zainstaluj wymagane zależności:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Przygotuj dane i wytrenuj model:**

   ```bash
   python src/main.py
   ```

5. **Uruchom aplikację konsolową:**

   ```bash
   python src/app.py
   ```

## Podział pracy

Prace nad projektem podzielone są według etapów

### 1. Preprocessing danych

**Odpowiedzialność:** Janulo

### 2. Wektoryzacja (TF-IDF)

**Odpowiedzialność:** higar

### 3. Wybór modelu

**Odpowiedzialność:** [@moshenetsb](https://github.com/moshenetsb)

### 4. Ewaluacja i interpretacja

**Odpowiedzialność:** [@shang1410](https://github.com/shang1410)

### Koordynatorka: Aleksandra
