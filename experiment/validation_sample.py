import csv
import json
import os


class ValidationSampler:
    """
    Zieht die geschichtete Validierungsstichprobe für den menschlichen
    Goldstandard (Abschnitt "Evaluierungsverfahren" der Thesis):

        - 24 der 240 Antworten,
        - jede der acht Artikelvarianten kommt in Kombination mit jedem der
          drei Presseartikel genau einmal vor (8 x 3 = 24 Zellen),
        - die beiden Modelle sind mit je zwölf Antworten balanciert vertreten,
        - die Seeds rotieren systematisch über die Zellen.

    Die Zuordnung ist deterministisch (Schachbrett-Muster über Artikel- und
    Methodenindex, Seed-Rotation über den Zellenindex) und damit vollständig
    reproduzierbar und auditierbar -- ein Zufallsgenerator ist nicht nötig.
    """

    def __init__(
        self,
        results_dir="../results",
        key_messages_path="../data/annotations/key_messages.json",
    ):
        self.results_dir = results_dir
        with open(key_messages_path, encoding="utf-8") as f:
            self.key_messages = json.load(f)

    # ------------------------------------------------------------------
    # Stichprobenziehung
    # ------------------------------------------------------------------

    def draw(self, output_file="validation_sample.csv", sheets_dirname="kodierboegen"):
        """
        Zieht die Stichprobe, prüft die Schichtungsbedingungen und schreibt:
            - results/validation_sample.csv        (Schlüssel + Antworttexte)
            - results/rating_sheet_rater1.csv      (leere Kodiervorlage)
            - results/rating_sheet_rater2.csv      (leere Kodiervorlage)
            - results/kodierboegen/*.md            (ein Kodierbogen je Antwort)
        """
        answers = self._load_answers()
        index = {self._key(row): row for row in answers}

        articles = sorted({row["Article_ID"] for row in answers})
        methods = sorted({row["Method"] for row in answers})
        models = sorted({row["Model"] for row in answers})
        seeds = sorted({str(row["Seed"]) for row in answers})
        if len(models) != 2:
            raise ValueError(f"Erwartet genau 2 Modelle, gefunden: {models}")

        sample = []
        for article_idx, article_id in enumerate(articles):
            for method_idx, method in enumerate(methods):
                model = models[(article_idx + method_idx) % 2]
                cell_idx = article_idx * len(methods) + method_idx
                seed = seeds[cell_idx % len(seeds)]
                key = (article_id, method, model, seed)
                if key not in index:
                    raise ValueError(f"Keine Experiment-Antwort zu Stichproben-Schlüssel {key}.")
                sample.append(index[key])

        self.verify(sample, articles, methods, models)

        sample_path = os.path.join(self.results_dir, output_file)
        self._write_sample_csv(sample, sample_path)
        for rater in (1, 2):
            self._write_rating_sheet(
                sample, os.path.join(self.results_dir, f"rating_sheet_rater{rater}.csv")
            )
        sheets_dir = os.path.join(self.results_dir, sheets_dirname)
        self._write_coding_sheets(sample, sheets_dir)

        print(
            f"Validierungsstichprobe gezogen: {len(sample)} Antworten -> {sample_path}\n"
            f"Kodiervorlagen: rating_sheet_rater1.csv / rating_sheet_rater2.csv\n"
            f"Kodierbögen: {sheets_dir}/"
        )
        return sample

    @staticmethod
    def verify(sample, articles, methods, models):
        """Prüft die Schichtungsbedingungen; wirft ValueError bei Verletzung."""
        if len(sample) != len(articles) * len(methods):
            raise ValueError(f"Stichprobe hat {len(sample)} statt {len(articles) * len(methods)} Zellen.")
        cells = {(row["Article_ID"], row["Method"]) for row in sample}
        if len(cells) != len(sample):
            raise ValueError("Mindestens eine Artikel-Methoden-Zelle ist doppelt besetzt.")
        model_counts = {model: 0 for model in models}
        for row in sample:
            model_counts[row["Model"]] += 1
        counts = sorted(model_counts.values())
        if counts[0] != counts[-1]:
            raise ValueError(f"Modelle nicht balanciert: {model_counts}")

    # ------------------------------------------------------------------
    # Ausgabedateien
    # ------------------------------------------------------------------

    def _write_sample_csv(self, sample, path):
        columns = ["Nr", "Article_ID", "Article_Type", "Method", "Seed", "Model", "Answer"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            for nr, row in enumerate(sample, start=1):
                writer.writerow({"Nr": nr, **row})

    def _write_rating_sheet(self, sample, path):
        """Leere Kodiervorlage; KM-Spalten entsprechend der Kernaussagen-Anzahl."""
        max_km = max(len(v["kernaussagen"]) for v in self.key_messages.values())
        columns = [
            "Nr", "Article_ID", "Method", "Seed", "Model",
            "Claims_Korrekt", "Claims_Intrinsisch", "Claims_Extrinsisch",
        ] + [f"KM_{i}" for i in range(1, max_km + 1)]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
            writer.writeheader()
            for nr, row in enumerate(sample, start=1):
                km_count = len(self.key_messages[row["Article_ID"]]["kernaussagen"])
                entry = {
                    "Nr": nr,
                    "Article_ID": row["Article_ID"],
                    "Method": row["Method"],
                    "Seed": row["Seed"],
                    "Model": row["Model"],
                }
                # Nicht existierende Kernaussagen werden mit "-" gesperrt.
                for i in range(1, max_km + 1):
                    entry[f"KM_{i}"] = "" if i <= km_count else "-"
                writer.writerow(entry)

    def _write_coding_sheets(self, sample, sheets_dir):
        os.makedirs(sheets_dir, exist_ok=True)
        for nr, row in enumerate(sample, start=1):
            article_id = row["Article_ID"]
            key_messages = self.key_messages[article_id]["kernaussagen"]
            numbered = "\n".join(f"{i}. {km}" for i, km in enumerate(key_messages, start=1))
            content = (
                f"# Kodierbogen {nr:02d}\n\n"
                f"| Feld | Wert |\n|---|---|\n"
                f"| Artikel | {article_id} ({self.key_messages[article_id].get('titel', '')}) |\n"
                f"| Variante | {row['Method']} |\n"
                f"| Modell | {row['Model']} |\n"
                f"| Seed | {row['Seed']} |\n\n"
                f"**Quelltext:** `data/processed/{article_id}_{row['Method']}.html` "
                f"(exakt die Artikelvariante, die das Modell im Experiment erhalten hat)\n\n"
                f"## Kernaussagen (je: abgedeckt ja/nein)\n\n{numbered}\n\n"
                f"## Zählfelder (gemäß Codebuch)\n\n"
                f"- Claims korrekt: ____\n- Claims intrinsisch fehlerhaft: ____\n"
                f"- Claims extrinsisch: ____\n\n"
                f"## Zu bewertende LLM-Antwort\n\n{row['Answer']}\n"
            )
            filename = f"{nr:02d}_{article_id}_{row['Method']}_{row['Model']}_seed{row['Seed']}.md"
            with open(os.path.join(sheets_dir, filename), "w", encoding="utf-8") as f:
                f.write(content)

    # ------------------------------------------------------------------
    # Datenzugriff
    # ------------------------------------------------------------------

    def _load_answers(self):
        answers = []
        for filename in sorted(os.listdir(self.results_dir)):
            if (
                filename.startswith("experiment_results_run")
                and filename.endswith(".csv")
                and "_DRYRUN" not in filename
            ):
                path = os.path.join(self.results_dir, filename)
                with open(path, newline="", encoding="utf-8") as f:
                    answers.extend(csv.DictReader(f, delimiter=";"))
        if not answers:
            raise FileNotFoundError(f"Keine Experiment-Ergebnisse in {self.results_dir} gefunden.")
        return answers

    @staticmethod
    def _key(row):
        return (row["Article_ID"], row["Method"], row["Model"], str(row["Seed"]))
