# Genetic Algorithm Optimizer

Graficzna aplikacja w Pythonie 3.12 do optymalizacji funkcji wielu zmiennych metodą algorytmu genetycznego.

## Funkcje

- binarna reprezentacja chromosomu z konfiguracją dokładności i liczby zmiennych
- selekcja: najlepszych, najgorszych, ruletka, turniejowa
- krzyżowanie: jednopunktowe, dwupunktowe, jednorodne, ziarniste
- mutacja: brzegowa, jednopunktowa, dwupunktowa
- inwersja z konfigurowalnym prawdopodobieństwem
- strategia elitarna jako procent lub liczba osobników
- zapis wyników kolejnych epok do SQLite
- trzy osobne wykresy: best value, average, std
- eksport CSV z chromosomami oraz PNG z wykresami dla każdego uruchomienia

## Uruchomienie

1. Zainstaluj zależności: `python -m pip install -r requirements.txt`
2. Uruchom aplikację: `python main.py`

## Uwagi

- Domyślnie dostępne są funkcje: Schwefel, Griewank.
- Wyniki są zapisywane do `data/results.sqlite3`.
- Eksportowane pliki trafiają do `data/exports/run_<id>/`.
