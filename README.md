# Porównanie systemów decyzyjnych – Fuzzy Logic vs Uczenie Maszynowe

## 1. Założenia

* **Zbiór danych:** Wykorzystano proponowany w zadaniu zbiór danych dotyczący diagnozy cukrzycy (`diabetes.csv`) zawierający 9 kolumn (w tym wynik) i ponad 700 rekordów.
* **Czyszczenie danych:** Dane surowe posiadały braki ukryte pod wartością "0" (np. 374 brakujących wyników dla insuliny), które najpierw zamieniono na wartości puste (NaN). W reprezentatywnej próbie badawczej zdecydowaliśmy się na usunięcie wszystkich wierszy z brakującymi wartościami, pozostawiając 392 rekordy.
* **Model Fuzzy Logic:** Zbudowano system oparty na logice rozmytej z wykorzystaniem biblioteki `scikit-fuzzy`, wybór atrybutów nastąpił zgodnie z wynikami macierzy korelacji, zatem atrybuty najbardziej skorelowane z występowaniem choroby, definiując trzy kluczowe: wiek (age), BMI oraz poziom glukozy (glucose).
* **Modele Uczenia Maszynowego (ML):** Do porównania wytypowano trzy modele z biblioteki `scikit-learn`: Decision Tree, Logistic Regression oraz Random Forest.
* **Metodologia oceny:** Modele oceniano przy użyciu 5-krotnej walidacji krzyżowej (Stratified 5-Fold Cross-Validation), następnie wybierając średnią ze wszystkich wyników dla każdej z metryk. Metryki sprawdzane dla każdego z tych modelów to następująco: Accuracy (Dokładność), Recall (Czułość), Precision (Precyzja) oraz F1-score

## 2. Opis przeprowadzonych testów

* **Fuzzy Logic:** Skonfigurowano różną ilość (3-4) oraz różne typy funkcji przynależności (trapezoidalne `trapmf` i trójkątne `trimf`) dla wejść, definiując m.in. stany poziomu glukozy (niski, średni, wysoki, alarmujący) oraz kategorie BMI i wieku. Reguły decyzyjne mapujące wejścia na szanse wystąpienia cukrzycy (od niskiej do pewnej) zostały napisane ręcznie, wzorując się przede wszystkim macierzą korelacji (bardziej skorelowane z wynikiem atrybuty mają większy wpływ na wyjście). Jako próg odcięcia dla klasyfikacji binarnej, w celu porównania metryk z resztą testów, w systemie rozmytym przyjęto wartość 55.0 (poniżej próg interpretowany jako 0, powyżej jako 1).
* **ML:** Równolegle przetestowano działanie algorytmów klasycznych (Decision Tree, Logistic Regression, Random Forest) dostępnych z biblioteki `scikit-learn` na dokładnie tak samo podzielonym i przetworzonym zbiorze danych. Obliczono dla każdego z modeli średnie wartości zaplanowanych metryk.

## 3. Wyniki

Poniżej znajduje się bezpośrednie zestawienie średnich wartości metryk uzyskanych w testach na próbie badawczej (z usuniętymi wierszami, które posiadały brakujące wartości):

| Model | Accuracy (Dokładność) | Recall (Czułość) | Precision (Precyzja) | F1-Score |
| --- | --- | --- | --- | --- |
| **Random Forest** | 0.7934 | 0.7615 | 0.6674 | 0.7066 |
| **Logistic Regression** | 0.7806 | 0.7462 | 0.6475 | 0.6906 |
| **Fuzzy Logic** | 0.7806 | 0.7000 | 0.6618 | 0.6795 |
| **Decision Tree** | 0.7373 | 0.8615 | 0.5741 | 0.6862 |

## 4. Dyskusja

* Zestawienie wykazuje, że model **Random Forest** osiągnął najlepszy ogólny rezultat, przewyższając pozostałe modele pod kątem dokładności (0.7934) i miary F1 (0.7066).
* Metoda **Fuzzy Logic** uzyskała dokładnie taką samą wartość Accuracy jak regresja logistyczna (0.7806), jednak odznaczyła się wyższą precyzją (0.6618 w stosunku do 0.6475), kosztem niższej czułości (0.7000 dla Fuzzy, 0.7462 dla Logistic Regression). Wynik ten udowadnia, że przy odpowiednim (nawet ręcznym) doborze 3 atrybutów oraz reguł, model rozmyty może rywalizować z algorytmami ML.
* **Decision Tree** wyróżniło się ekstremalnie wysokim wskaźnikiem Recall (0.8615), co w przypadku medycyny jest pożądaną cechą (rzadko omija rzeczywiste przypadki chorobowe), jednak kosztem dużej liczby fałszywych alarmów, o czym świadczy najniższa ze wszystkich precyzja (0.5741).

## 5. Wnioski

1. Systemy oparte na sieciach decyzyjnych (ML) oferują wyższą gotowość do predykcji prosto po stronie kodu, ze względu na m.in. brak potrzeby manualnego układania dziesiątek reguł, prostota w tworzeniu takowych systemów, ze względu na dostępność różnych bibliotek.W przypadku bardziej zaawansowanych modeli (Random Forest) osiągają nieco lepsze ogólne wyniki niż proste systemy rozmyte.
2. Zaletą użycia Fuzzy Logic w diagnostyce jest przede wszykim wyjaśnialność. Każda podjęta decyzja wynika wprost ze zdefiniowanej funkcji opartej na zapisanych regułach (np. "jeżeli glukoza alarmująca i osoba otyła, szansa na cukrzycę: pewna"). Dla tej metody możemy również bezpośrednio wybrać atrybuty, które będziemy rozpatrywali, co może pozwolić na uniknięcie overfittingu dla zestawów danych z większą ilością atrybutów, gdzie część z nich bezpośrednio wpływa na siebie.
3. Zaprojektowany ewaluator pokazał, że dla tego specyficznego zbioru, najlepiej zbilansowanym i sprawdzonym rozwiązaniem ML jest algorytm Random Forest. Podejście oparte na Fuzzy Logic to dobra alternatywa, jednak dla bardziej zaawansowanych, np. medycznych zbiorów, tak jak ten, znacznie lepiej sprawdza się, gdy reguły decyzyjne lub funckje przynależnosci zapisywane są przez osobę mającą wiedzę z określonej dziedziny.
