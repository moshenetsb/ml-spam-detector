from pathlib import Path

import pandas as pd
import numpy as np 

#wczytanie danych
base_path = Path(__file__).resolve().parent.parent

spam = pd.read_csv(base_path / 'data/data_raw', sep='\t', header=None, names=['label', 'text'])

print(spam.head()) #podgląd danych

def get_percentage_missing(df, axis):
  """
    Zwraca procent brakujących wartości (NaN) w wierszach lub kolumnach DataFrame.

    df : pandas.DataFrame
        Tabela danych, dla której liczone są braki.

    axis : int
        Oś, względem której obliczane są braki:
        - 0 : procent braków w każdej kolumnie
        - 1 : procent braków w każdym wierszu
  """
  missing_rows = df.isna().sum(axis=axis)
  missing_rows_perc = np.round(missing_rows / df.shape[axis] * 100, 2)
  return missing_rows_perc[missing_rows_perc > 0]

braki_w_kolumnach = get_percentage_missing(spam, axis=0)
print(braki_w_kolumnach)

missing_rows_ti = get_percentage_missing(spam, axis=1)

missing_all_column_values = missing_rows_ti[missing_rows_ti == 100].index
print(missing_all_column_values) # sprawdze czy nie ma pustych wierszy 
print(spam.duplicated().sum()) #ilość zduplikowanych wierszy

duplikaty_spam = spam[spam.duplicated(keep=False)] 

#print(duplikaty_spam.sort_values(by='text').head(4))
#pokazanie duplikatów
 

spam = spam.drop_duplicates()
#usuniecie duplikatów

print(spam.duplicated().sum())
#sprawdzenie ilości duplikatów

katalog_docelowy = Path("data")

katalog_docelowy.mkdir(parents=True, exist_ok=True)

spam.to_csv(katalog_docelowy / 'data_processed.csv', index=False)
#tworzenie pliku z gotowymi danymi