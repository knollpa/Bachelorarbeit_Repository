# GEO im BMW Group Newsroom — Code zur Bachelorarbeit

Dieses Repository enthält den vollständigen Programmcode zur Bachelorarbeit
**„Generative Engine Optimization (GEO) für das Presseportal der BMW Group:
Validierung von Optimierungsmethoden und Entwicklung eines Redaktionsleitfadens"**

Die Arbeit untersucht in einem kontrollierten Experiment, ob sieben literaturbasiert
ausgewählte GEO-Methoden die Qualität beeinflussen, mit der Sprachmodelle
Presseartikel wiedergeben. Gemessen wird über drei Metriken: Information Accuracy,
Hallucination Rate und Key Message Coverage.

## ⚠️ Nicht enthaltene Daten

Die untersuchten Presseartikel stammen aus dem zum Zeitpunkt der Arbeit noch
unveröffentlichten BMW Group Newsroom und unterliegen einem Sperrvermerk.
**Die Verzeichnisse `data/` und `results/` sind deshalb nicht Bestandteil dieses
Repositories.** Das betrifft die Rohartikel, die daraus erzeugten Varianten, die
Modellantworten, die Bewertungsdateien und die Kodierbögen der menschlichen Rater.

Daraus folgt:

- Die Pipeline lässt sich **ohne diese Daten nicht ausführen**. Der Code dient der
  Nachvollziehbarkeit des Verfahrens, nicht der Wiederholung des Experiments.

## Pipeline im Überblick

```
data/raw/            Rohartikel (HTML)
      │
      ▼  experiment/data_loader.py          Bereinigung, Extraktion des JSON-LD-Blocks
data/processed/      Baseline + 7 Varianten je Artikel
      │
      ▼  transform/                          Erzeugung der Varianten (Claude Opus 4.8)
      ▼  transform/verification.py           Manipulation Check der Varianten
      │
      ▼  experiment/experiment_runner.py     240 Modellantworten (GPT-5.4 mini, Gemini 3.5 Flash)
results/experiment_results_run*.csv
      │
      ▼  experiment/judge.py                 LLM-as-a-Judge (Claude Opus 4.8)
results/judge_results.csv
      │
      ▼  experiment/analysis.py              Statistische Auswertung, Tabellen und Abbildungen
      ▼  experiment/agreement.py             Übereinstimmung mit dem menschlichen Goldstandard
```

## Zuordnung Modul - Abschnitt der Arbeit

| Modul | Aufgabe | Abschnitt |
|---|---|---|
| `experiment/data_loader.py` | Bereinigung der Rohartikel | 4.2.2 Aufbereitung und Dateistruktur |
| `transform/prompts.py` | Transformationsprompts der sieben Methoden | 4.3.2, Anhang A.4 |
| `transform/variant_generator.py`, `transform/cli.py` | Erzeugung der 21 Varianten | 4.3.2 |
| `transform/verification.py` | Manipulation Check (Schritt 1 der Verifikation) | 4.3.3 |
| `experiment/llm_connector.py` | Kapselung der OpenAI- und Google-Aufrufe | 4.4.1 |
| `experiment/experiment_runner.py` | Ablaufsteuerung der fünf Durchläufe | 4.4.1, 4.4.2 |
| `experiment/judge.py` | LLM-as-a-Judge, Metrikberechnung | 5.1 |
| `experiment/validation_sample.py` | Geschichtete Validierungsstichprobe, Kodierbögen | 5.1 |
| `experiment/agreement.py` | Krippendorffs α, Balanced Accuracy | 5.1 |
| `experiment/analysis.py` | Wilcoxon, Holm, Effektstärken, Spearman | 5.1 bis 5.3 |

## Setup

```bash
poetry install
```

Alternativ mit pip:

```bash
pip install -e ".[dev]"
```

Für die Ausführung der Pipeline werden folgende
Umgebungsvariablen benötigt, üblicherweise in einer `.env`-Datei:

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Notebooks

`notebooks/llm_experiment.ipynb` und `notebooks/judge_evaluation.ipynb` dokumentieren
die tatsächliche Ausführung von Experiment und Bewertung. Die gespeicherten Ausgaben
enthalten Dateilistings und Fortschrittsprotokolle, keine Artikelinhalte und keine
Modellantworten.

