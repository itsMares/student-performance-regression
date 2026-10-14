# Vorhersage von Schülerleistungen mit neuronalen Netzen (Regression)

**Data-Science-Projekt von Mares Büchler** | Wirtschaftsinformatik (B.Sc.), HAW Kiel

Ein neuronales Netz sagt den Gesamtscore (`overall_score`, 0 bis 100 Punkte) von 25.000 Schülerinnen und Schülern vorher. Grundlage sind ausschließlich früh verfügbare Merkmale wie Lernzeit, Anwesenheit oder Bildungshintergrund der Eltern. Das Projekt führt von einer einfachen Baseline über ein tiefes Netz und eine Hyperparameteroptimierung zu einem finalen Modell, das einmalig auf einem unberührten Testset bewertet wird.

**[Zum Notebook](KI_ML_student_performance_REG_Mares_Buechler.ipynb)**: Alle Ausgaben und Grafiken sind gespeichert und direkt auf GitHub sichtbar, ohne dass das Notebook ausgeführt werden muss.

---

## Projektkontext

Das Notebook ist im Sommersemester 2026 in der Modulreihe *KI: Machine Learning* und *KI: Deep Learning* im Studiengang Wirtschaftsinformatik (B.Sc.) an der HAW Kiel entstanden.

Die Prüfungsleistung war eine Partnerarbeit mit dem gemeinsamen Ziel eines datenbasierten Frühwarnsystems für leistungsgefährdete Studierende. Sie bestand aus drei Teilanalysen auf demselben Datensatz:

| Teil | Fragestellung | Umsetzung | In diesem Repo |
|---|---|---|---|
| **Regression** | Wie hoch fällt die Leistung aus? | Mares Büchler | enthalten |
| Klassifikation | Welche Risikoklasse liegt vor? | weiteres Gruppenmitglied | nicht enthalten |
| Clustering | Welche Lerntypen zeigen sich? | weiteres Gruppenmitglied | nicht enthalten |

Dieses Repository enthält den von mir eigenverantwortlich umgesetzten **Regressionsteil**. Dazu gehört der gemeinsame Vorverarbeitungsteil (im Notebook „Master-Teil“), der allen drei Analysen als Ausgangslage diente. Er baut auf einer vorangegangenen Regressionsanalyse von mir aus dem Modul *Grundlagen der Datenanalyse* auf.

Weil der Master-Teil für alle drei Analysen gilt, erwähnt das Notebook an einigen Stellen auch die Klassifikation und das Clustering. Diese Verweise dienen nur der Einordnung. Die Analyse selbst ist vollständig und eigenständig nachvollziehbar.

---

## Repository-Struktur

```
├── KI_ML_student_performance_REG_Mares_Buechler.ipynb   # Notebook mit allen Ausgaben
├── Student_Performance.csv                              # Datensatz (Quelle siehe unten)
└── README.md
```

---

## Vorgehen

1. **Datenaufbereitung:** Entfernen von Spalten mit direktem Bezug zur Zielgröße (Teilnoten, Endnote), um Data Leakage zu vermeiden. Danach Ordinal Encoding für Merkmale mit Rangfolge und One-Hot-Encoding für alle übrigen kategorialen Merkmale.
2. **Explorative Analyse:** Prüfung auf fehlende Werte und Ausreißer, Korrelationsanalyse mit Heatmap.
3. **Saubere Datentrennung:** 3-Wege-Split in 64 % Training, 16 % Validierung und 20 % Test. Der Scaler wird nur auf den Trainingsdaten gefittet. Jede Modellauswahl erfolgt auf der Validierung, das Testset bleibt bis zum Schluss unberührt.
4. **Baseline:** Kleines Netz mit einer verborgenen Schicht als Referenzpunkt.
5. **Tiefes Netz:** Vier verborgene Schichten unter identischem Trainingsprotokoll, um den reinen Effekt der Modellkapazität zu messen.
6. **Hyperparameteroptimierung:** Manueller Grid-Search über 4 Architekturen × 2 Lernraten, danach ein Gegentest mit Dropout (verworfen, da er den Validierungsfehler erhöht).
7. **Finales Modell:** Refit auf den vollen 80 % Trainingsdaten und eine einmalige Bewertung auf dem Testset.
8. **Absicherung der Ergebnisse:** Vergleich mit einem Ein-Feature-Modell (Feature-Ceiling) und Permutation Importance.

Ein globaler Seed sorgt dafür, dass alle Ergebnisse reproduzierbar sind.

---

## Ergebnisse

**Finales Modell auf dem Testset** (5.000 ungesehene Datensätze):

| Metrik | Wert |
|---|---|
| MAE | 4,93 Punkte |
| RMSE | 5,83 Punkte |
| R² | 0,906 |

Das Modell erklärt damit gut 90 % der Varianz im Gesamtscore und liegt im Schnitt knapp 5 Punkte neben dem tatsächlichen Wert.

**Modellvergleich auf dem Validierungsset:**

| Modell | Parameter | MAE | R² |
|---|---|---|---|
| Baseline (1 Hidden Layer, 32 Units) | 545 | 4,871 | 0,9090 |
| Tiefes Netz (4 Hidden Layer, 128-64-32-16) | 12.929 | 4,658 | 0,9132 |
| Gewinner Grid-Search (128-64-32-16, LR 0,0005) | 12.929 | 4,715 | 0,9115 |

**Zentrale Erkenntnis: Die Grenze liegt in den Daten, nicht im Modell.**
Ein einfaches lineares Modell mit nur einem Merkmal (`study_hours`) erreicht auf dem Testset bereits R² = 0,82. Alle 14 weiteren Merkmale und ein tiefes Netz mit rund 24-mal so vielen Parametern heben das R² nur auf 0,906. Die Permutation Importance bestätigt das: Wird `study_hours` zufällig durchmischt, steigt der MAE um rund 15,6 Punkte. Bei allen anderen Merkmalen außer der Anwesenheit (+3,2) sind es höchstens 0,34 Punkte.

---

## Notebook lokal ausführen

Verwendet wurden Python 3.12 und TensorFlow 2.21.

```bash
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib jupyter
jupyter notebook KI_ML_student_performance_REG_Mares_Buechler.ipynb
```

- Die CSV-Datei muss im selben Ordner wie das Notebook liegen.
- Der Grid-Search trainiert acht Modelle nacheinander und kann einige Minuten dauern.
- Die grafische Netzdarstellung (`plot_model`) benötigt optional Graphviz. Fehlt es, wird die Grafik übersprungen, und das Notebook läuft trotzdem durch.

## Verwendete Python-Pakete

| Paket | Einsatz im Projekt |
|---|---|
| pandas, NumPy | Einlesen, Aufbereitung und Berechnung der Daten |
| scikit-learn | Train-Test-Split, Skalierung, lineares Vergleichsmodell, Gütemetriken |
| TensorFlow / Keras | Aufbau, Training und Auswertung der neuronalen Netze |
| Matplotlib, Seaborn | Boxplots, Heatmap, Lernkurven und Diagnoseplots |
| Jupyter | Ausführung und Dokumentation im Notebook |

---

## Datensatz und Lizenz

Der Datensatz **„Student Performance Dataset“** stammt von **kundanbedmutha** auf Kaggle:
https://www.kaggle.com/datasets/kundanbedmutha/student-performance-dataset

Er steht unter der Lizenz **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**. Die Datei `Student_Performance.csv` liegt hier unverändert vor. Alle Bearbeitungsschritte (Entfernen von Spalten, Encoding, Skalierung) finden ausschließlich im Notebook statt.

## Hinweis zur KI-Nutzung

Bei der Entstehung dieses Projekts wurde ein KI-Assistent unterstützend eingesetzt, unter anderem für Strukturierung, Fehlersuche und die LaTeX-Formatierung. Umfang und Eigenleistung sind am Ende des Notebooks transparent beschrieben.

---

**Mares Büchler**, Wirtschaftsinformatik (B.Sc.), HAW Kiel
