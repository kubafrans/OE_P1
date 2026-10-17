# Genetic Algorithm Optimizer

Graficzna aplikacja w Pythonie 3.11 do minimalizacji i maksymalizacji funkcji wielu zmiennych klasycznym binarnym algorytmem genetycznym.

## Funkcje

- binarna reprezentacja chromosomu z konfiguracją dokładności i liczby zmiennych
- selekcja: najlepszych, najgorszych, ruletka, turniejowa
- krzyżowanie: jednopunktowe, dwupunktowe, jednorodne, ziarniste
- mutacja: brzegowa, jednopunktowa, dwupunktowa
- inwersja z konfigurowalnym prawdopodobieństwem
- strategia elitarna jako procent lub liczba osobników
- zapis wyników kolejnych epok do SQLite
- trzy osobne wykresy: best value, average, std
- eksport CSV z metrykami każdej generacji, JSON z ustawieniami i wynikiem, PNG z wykresami
- powtórzenia z różnymi ziarnami: średnia, najlepszy i najgorszy wynik oraz średni czas
- CEC 2014 F9 (Shifted and Rotated Rastrigin) przez bibliotekę `opfunu`

## Uruchomienie

Wymagany Python **3.11** z Tkinter i środowisko graficzne. `opfunu==1.0.4` deklaruje Python `<3.12`.
Do utworzenia środowiska użyj Pythona 3.11.

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv311\Scripts\python.exe main.py
```

`requirements-lock.txt` zawiera dokładne wersje zależności użyte do weryfikacji aplikacji. `requirements.txt` określa bezpośrednie zależności projektu. Środowisko wirtualne należy utworzyć lokalnie po sklonowaniu repozytorium.

## Funkcje i konfiguracja GUI

| Funkcja | Zakres | Liczba zmiennych |
|---|---|---|
| Schwefel | [-500, 500] | Dowolna dodatnia liczba |
| Griewank | [-600, 600] | Dowolna dodatnia liczba |
| CEC 2014 F9 | [-100, 100] | 10, 20, 30, 50, 100 |

Wybór funkcji ustawia jej standardowe granice. Dla CEC ustawia także 10 zmiennych; można zmienić na inny obsługiwany wymiar. Funkcje podstawowe nadal obsługują np. 5, 10, 20 i 27 zmiennych. Granice CEC są walidowane, aby zachować benchmark z PDF.

`precision` oznacza maksymalny odstęp siatki dekodowania, a nie liczbę miejsc po przecinku. Wszystkie kody binarne odwzorowują różne równomiernie rozmieszczone wartości pomiędzy granicami. Dokładne optimum ciągłe może nie leżeć na tej siatce.

Ustaw `repetitions` na **10 lub więcej**, aby otrzymać statystyki do sprawozdania. `base seed` ustala pierwsze ziarno; kolejne uruchomienia używają kolejnych liczb. Przy jednym powtórzeniu GUI pokazuje postęp generacji; przy wielu postęp powtórzeń i wykresy najlepszego uruchomienia.

Prawdopodobieństwa mutacji i inwersji dotyczą potomka, a krzyżowania pary rodziców. `percentage best to select` określa wielkość puli rodziców dla każdej metody selekcji. Turnieje mają co najwyżej rozmiar populacji. Ziarniste krzyżowanie przełącza źródło bloków długości `max(1, długość chromosomu // 4)` z prawdopodobieństwem 0.5. Generacja 0 to populacja początkowa; N epok daje N+1 rekordów wraz z populacją końcową.

## Zapis wyników

Każde uruchomienie GUI zapisuje metryki generacji w SQLite oraz eksportuje CSV i JSON z konfiguracją. Wykresy są eksportowane jako PNG. Przy wielu powtórzeniach aplikacja zapisuje również zestawienie wyników i wykresy najlepszego uruchomienia. Pliki można wykorzystać do ręcznego opracowania sprawozdania.

Czas obejmuje inicjalizację populacji i obliczenia GA, bez ładowania funkcji, zapisu i eksportu wykresów. Błąd względem znanego minimum jest wyliczany tylko dla minimalizacji.

## Uwagi

- Funkcje są dostarczane przez `benchmark_functions` i `opfunu`, bez własnej implementacji wzorów.
- Wyniki są zapisywane do `data/results.sqlite3`.
- Eksportowane pliki trafiają do `data/exports/run_<id>/`.
- Katalog `data/` powstaje automatycznie; bazy i wyniki uruchomień nie są przechowywane w Git.
- Dane przesunięcia i macierze CEC są dostarczane przez `opfunu`; plik `Cec2014.pdf` nie jest potrzebny do uruchomienia.
- Aby odtworzyć obliczenia, ustaw tę samą konfigurację i `base seed`. Wyniki liczbowe są powtarzalne w tym samym środowisku; czas zależy od sprzętu i obciążenia. Pełne ustawienia każdego uruchomienia są zapisywane w `run.json`.
