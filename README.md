# ha_stance_detection
Material für Hausarbeit im Kurs Stance Detection, Universität Potsdam, Sommersemester 2021

Replikationsprojekt für

Wojatzki, Michael und Torsten Zesch (2016). "Stance-based argument mining – modeling implicit argumentation using stance". In: Proceedings of the KONVENS, S. 313–322.


Die Daten, mit denen gearbeitet wurde, sind verfügbar unter:


https://github.com/muchafel/AtheismStanceCorpus

## Requirements
Python >= 3.8


weitere Abhängigkeiten sind:

* nltk
* sklearn
* pandas

Abhängigkeiten können mit dem folgenden Befehl installiert werden:
```
pip install requirements.txt
```

## Benutzung
Aufruf von der Kommandozeile mit
```
$ python main.py --help
usage: main.py [--help] data_path

Process and classify data for explicit and debate stance using a SVM in 10-fold cross validation and a decision tree classifier.

positional arguments:
  data_path   File to process

optional arguments:
  --help, -h  Show this help message an exit

```
## Autorin
Luise Köhler
: luise.koehler(a)uni-potsdam.de