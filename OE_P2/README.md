# Obliczenia ewolucyjne — projekt 2

Aplikacja graficzna do optymalizacji funkcji algorytmem genetycznym. Rozszerza projekt 1 o chromosomy rzeczywiste; zachowuje tryb binarny do porównań.

## Wymagania i uruchomienie

Python **3.11** z komponentem Tcl/Tk oraz środowisko graficzne. W PowerShell, w katalogu projektu:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv311\Scripts\python.exe main.py
```

Na stanowisku z gotowym środowiskiem OE_P1, z katalogu OE_P2:

```powershell
..\OE_P1\.venv311\Scripts\python.exe main.py
```

Nie używaj samego `python main.py`, jeśli aktywne środowisko nie ma zainstalowanych zależności. Projekt może działać samodzielnie po instalacji wymagań; korzystanie ze środowiska P1 jest opcją lokalną.

## Możliwości

- Reprezentacja `binary` z projektu 1 i `real` z projektu 2.
- Krzyżowanie rzeczywiste: arytmetyczne, linearne, BLX-α, BLX-α-β i uśredniające.
- Mutacja rzeczywista: równomierna lub Gaussa; zmienia jedną losową współrzędną.
- Selekcja najlepszych, najgorszych, ruletkowa i turniejowa oraz elityzm.
- Minimalizacja i maksymalizacja funkcji Schwefela, Griewanka i CEC 2014 F9.
- Powtórzenia z kolejnymi ziarnami, zapis SQLite oraz eksport CSV, JSON i PNG.

## Obsługa

Ustaw funkcję, liczbę zmiennych, populację, liczbę epok i kierunek optymalizacji. Na dole przewijanego panelu wybierz `representation`. Dla `real` użyj pól `real crossover` i `real mutation`. Następnie kliknij **Uruchom**.

W trybie rzeczywistym `precision`, inwersja i wybór operatorów binarnych nie wpływają na przebieg. `Gaussian sigma / range` określa odchylenie szumu jako część szerokości dziedziny. Wartości poza dziedziną są przycinane do jej granic. Krzyżowanie linearne tworzy trzy kandydatury, ocenia je i wybiera dwie najlepsze przed mutacją. BLX-α-β rozszerza przedział o α od strony lepszego rodzica i β od strony gorszego.

Krótka próba działania: populacja 40, epoki 30, powtórzenia 1, ziarno 42. Do porównań ustaw co najmniej 10 powtórzeń i zachowaj wspólne parametry oraz ziarna między wariantami. Jedna próba nie wystarcza do oceny przewagi operatora.

## Czytanie wyników

Przy minimalizacji niższy wynik jest lepszy. Minimum Griewanka wynosi 0, Schwefela około 0, a CEC 2014 F9 wynosi 900. Błąd bezwzględny to odległość wyniku od minimum referencyjnego. Dla CEC wynik 905 oznacza błąd 5.

- **Best**: najlepszy wynik w bieżącej populacji.
- **Average**: średnia wyników osobników w danej generacji.
- **Std**: odchylenie standardowe wyników osobników w danej generacji.

Generacja 0 to populacja początkowa. Małe Std oznacza podobne wyniki, ale nie gwarantuje bliskości minimum. Średnia i odchylenie wyników wielu powtórzeń to inne miary niż Average i Std pojedynczej populacji.

## Pliki projektu

`main.py` uruchamia GUI, a `src/ga_optimizer/` zawiera konfigurację, silnik, operatory, benchmarki, zapis wyników i interfejs. `requirements.txt` określa zależności bezpośrednie, `requirements-lock.txt` zapisuje ich dokładne wersje.

Katalog `data/` jest tworzony podczas pracy: baza to `data/results.sqlite3`, a eksporty trafiają do `data/exports/`. Wygenerowanych wyników, środowisk wirtualnych i archiwów ZIP nie trzeba dodawać do repozytorium.
