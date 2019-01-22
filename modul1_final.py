""" Moduł wczytujący plik .csv z danymi wejściowymi, sprawdzający standard formatu pliku oraz kompletność danych """

import pandas as pd
import numpy as np


""" Wczytanie pliku z danymi jako ramka danych (plik z dowolnym separatorem)"""
df = pd.read_csv("C:/Users/Tomasz/Desktop/ClusterProject/iris_coma.csv", sep="[;\t,]", engine="python")
# print(df.head())


def object_checking():
    """ Funkcja sprawdzająca czy w danych nie ma obiektów typu łańcuch znaków """
    object_value = df.select_dtypes(include='object')
    arr = np.array(object_value)

    if arr.dtype == 'object':
        print("Error: string-object in column: " + str(object_value.columns[0])+". Correct your data to a numeric type.")
    else:
        print("Correct data.")

object_checking()


# def comas_to_dots():
#     """ Funkcja konwertująca kropki na przecinki w liczbach zmiennoprzecinkowych """
#     df = pd.read_csv("C:/Users/Tomasz/Desktop/ClusterProject/iris_przecinki.csv", sep=";")
#     df.to_csv("iris_dot.csv", sep='\t', encoding='utf-8', decimal='.')



def header_checking():
    """ Funkcja sprawdzająca czy wszystkie kolumny mają nagłówki """

    headers = df.columns.str.contains('^Unnamed')
     
    if True in headers: 
        print("Warning! Missing values in data. Please, complete headers in your data.")
    else: 
        print("Your dataset is correct.")


header_checking()


def data_completeness():
    """ Funkcja sprawdzajająca czy wszystkie dane zostały uzupełnione """

    for key, value in df.iteritems():
    
        nan_value = value.hasnans
        if nan_value is True:
            print("Warning!", value.hasnans.sum(), "uncompleted data at:\n", df[df.isnull().T.any().T])
        elif 'object' in df.dtypes:
            print("Warning! String object")
        else:
            """" Eksport danych po standaryzacji do pliku (pamięci) wewnętrznego """
            df.to_csv('C:/Users/Tomasz/Desktop/ClusterProject/irisInner.csv', sep="\t", header=True)


data_completeness()


def variable_renaming(df):
    """ Funkcja sprawadzająca czy poszczególne zmienne  mają swoje nazwy. Jeśli tak - przypisuje nazwy windeksom wierszy i usuwa pierwszą kolumną z nazwami. Jeśli nie - numeruje je kolejno zaczynając od 1. """
    print("Czy dane w poszczególnych wierszach zbioru danych zmienne mają swoje nazwy? [t/n] ")
    var_names = str(input())
    df.index.names = ['Name']
    if var_names.lower() == "t":
        df = df.rename(df.iloc[:, 0])
        df = df.drop(list(df)[0], axis=1)
        return df
    else:
        n = df.shape[0]
        df.index = range(1, n+1)
        return df


print(variable_renaming(df))


# dodać opcję graficznego porównania ze sobą różnych metod i miar odległości
# poprawić projekt interfejsu
# https://stackoverflow.com/questions/5089030/how-do-i-create-a-radial-cluster-like-the-following-code-example-in-python