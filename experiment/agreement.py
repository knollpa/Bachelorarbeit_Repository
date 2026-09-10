import csv
import json
import os

import krippendorff
import numpy as np


class AgreementAnalyzer:
    """
    Quantifiziert die Übereinstimmung zwischen dem LLM-as-a-Judge-Evaluator
    und dem menschlichen Goldstandard (Abschnitt "Evaluierungsverfahren"):

        - je Metrik (IA, HR, KMC): Krippendorffs Alpha auf Intervallskala,
        - für die binären Kernaussagen-Urteile: zusätzlich Balanced Accuracy,
        - bei zwei Ratern: alle paarweisen Alphas (Judge-R1, Judge-R2, R1-R2
          als Mensch-Mensch-Referenz); der Goldstandard ist der Mittelwert
          beider Rater je Bewertungseinheit.

    Gate laut Thesis: Die automatisierten Urteile gelten erst ab
    Alpha >= 0,8 (je Metrik, Judge vs. Goldstandard) als hinreichend valide.

    Sonderfall fehlende Varianz (Abschnitt "Evaluierungsverfahren"): Weisen
    beide Kodierungen einer Metrik denselben konstanten Wert auf, ist Alpha
    mangels erwarteter Uneinigkeit nicht definiert. Dieser Fall kann
    konstruktionsbedingt nur bei vollstaendiger Uebereinstimmung eintreten;
    das Gate wird dann ueber die exakte Uebereinstimmungsquote (= 1,0)
    entschieden und diese ersatzweise berichtet.

    Praezisierung Quasi-Degeneration: Weicht hoechstens eine einzige
    Bewertungseinheit ueberhaupt zwischen den Kodierungen ab, ist Alpha zwar
    formal definiert, kollabiert aber wegen des Praevalenzparadoxons (Gwet
    2008) auf Werte nahe null und traegt keine Information ueber die
    tatsaechliche Uebereinstimmung. Das Gate wird dann ueber die exakte
    Uebereinstimmungsquote >= 0,95 entschieden und der Fall als degeneriert
    ausgewiesen.
    """

    ALPHA_GATE = 0.8
    EXACT_GATE_QUASI = 0.95
    METRICS = ("IA", "HR", "KMC")

    def __init__(self, results_dir="../results"):
        self.results_dir = results_dir

    # ------------------------------------------------------------------
    # Öffentliche Auswertung
    # ------------------------------------------------------------------

    def report(
        self,
        judge_file="judge_pilot.csv",
        rater1_file="rating_sheet_rater1.csv",
        rater2_file="rating_sheet_rater2.csv",
    ):
        """
        Erstellt den Übereinstimmungsbericht. Regulärer Pfad sind zwei
        unabhängige Rater; fehlt die Rater-2-Datei oder ist sie unausgefüllt,
        wird aus Robustheitsgründen der Einzelrater-Pfad verwendet.
        """
        judge = self._load_judge(os.path.join(self.results_dir, judge_file))
        rater1 = self._load_rating_sheet(os.path.join(self.results_dir, rater1_file))
        rater2 = self._try_load_rating_sheet(os.path.join(self.results_dir, rater2_file))

        keys = sorted(set(judge) & set(rater1))
        if not keys:
            raise ValueError("Keine gemeinsamen Bewertungseinheiten zwischen Judge und Rater 1.")
        if rater2 is not None:
            keys = sorted(set(keys) & set(rater2))

        print(f"Bewertungseinheiten: {len(keys)}  |  Rater: {'2 (Inter-Rater)' if rater2 else '1 (Einzelrater-Fallback)'}")
        results = {"n_units": len(keys), "two_raters": rater2 is not None, "metrics": {}}

        gate_passed = True
        for metric in self.METRICS:
            judge_values = [judge[k][metric] for k in keys]
            r1_values = [rater1[k][metric] for k in keys]
            coder_series = [judge_values, r1_values]
            row = {
                "alpha_judge_rater1": self._alpha_interval(judge_values, r1_values),
                "exact_judge_rater1": self._exact_agreement(judge_values, r1_values),
            }

            if rater2 is not None:
                r2_values = [rater2[k][metric] for k in keys]
                coder_series.append(r2_values)
                gold = [self._mean_or_none(a, b) for a, b in zip(r1_values, r2_values)]
                row["alpha_rater1_rater2"] = self._alpha_interval(r1_values, r2_values)
                row["alpha_judge_rater2"] = self._alpha_interval(judge_values, r2_values)
                row["exact_judge_rater2"] = self._exact_agreement(judge_values, r2_values)
                row["alpha_judge_gold"] = self._alpha_interval(judge_values, gold)
            else:
                gold = r1_values
                row["alpha_judge_gold"] = row["alpha_judge_rater1"]

            row["exact_agreement"] = self._exact_agreement(judge_values, gold)
            row["deviating_units"] = self._deviating_units(coder_series)
            if row["alpha_judge_gold"] is None:
                # Alpha ist genau dann nicht berechenbar, wenn saemtliche
                # paarbaren Werte identisch sind (fehlende Varianz) -- das
                # setzt vollstaendige Uebereinstimmung voraus.
                row["degenerate"] = True
                row["gate"] = row["exact_agreement"] == 1.0
            elif row["alpha_judge_gold"] < self.ALPHA_GATE and row["deviating_units"] <= 1:
                # Quasi-Degeneration: Die gesamte Varianz stammt aus einer
                # einzigen Bewertungseinheit; Alpha kollabiert wegen des
                # Praevalenzparadoxons und ist nicht aussagekraeftig.
                row["degenerate"] = True
                row["gate"] = row["exact_agreement"] >= self.EXACT_GATE_QUASI
            else:
                row["degenerate"] = False
                row["gate"] = row["alpha_judge_gold"] >= self.ALPHA_GATE
            gate_passed = gate_passed and row["gate"]
            results["metrics"][metric] = row

        # Balanced Accuracy für die binären Kernaussagen-Urteile
        results["km_balanced_accuracy"] = self._km_balanced_accuracy(judge, rater1, rater2, keys)

        self._print_report(results)
        return results

    # ------------------------------------------------------------------
    # Kennzahlen
    # ------------------------------------------------------------------

    @staticmethod
    def _alpha_interval(values_a, values_b):
        """Krippendorffs Alpha (Intervallskala) für zwei Kodierer; None-tolerant."""
        data = [
            [np.nan if v is None else float(v) for v in values_a],
            [np.nan if v is None else float(v) for v in values_b],
        ]
        try:
            return round(
                float(krippendorff.alpha(reliability_data=data, level_of_measurement="interval")),
                4,
            )
        except (ValueError, ZeroDivisionError):
            return None

    def _km_balanced_accuracy(self, judge, rater1, rater2, keys):
        """Balanced Accuracy der Judge-KM-Urteile gegen den menschlichen Standard."""
        report = {}
        for label, rater in (("rater1", rater1), ("rater2", rater2)):
            if rater is None:
                continue
            judge_flags, human_flags = [], []
            for key in keys:
                judge_km = judge[key]["km"]
                human_km = rater[key]["km"]
                for km_id, human_value in human_km.items():
                    if km_id in judge_km and human_value is not None:
                        judge_flags.append(judge_km[km_id])
                        human_flags.append(human_value)
            report[label] = self._balanced_accuracy(human_flags, judge_flags)
        return report

    @staticmethod
    def _balanced_accuracy(truth, predicted):
        pairs = list(zip(truth, predicted))
        positives = [(t, p) for t, p in pairs if t]
        negatives = [(t, p) for t, p in pairs if not t]
        if not positives or not negatives:
            return None
        tpr = sum(1 for _, p in positives if p) / len(positives)
        tnr = sum(1 for _, p in negatives if not p) / len(negatives)
        return round((tpr + tnr) / 2, 4)

    @staticmethod
    def _deviating_units(coder_series):
        """Anzahl der Einheiten, in denen die Kodierungen nicht identisch sind."""
        count = 0
        for values in zip(*coder_series):
            present = {v for v in values if v is not None}
            if len(present) > 1:
                count += 1
        return count

    @staticmethod
    def _exact_agreement(values_a, values_b):
        """Anteil exakt uebereinstimmender Wertepaare (None-tolerant)."""
        pairs = [
            (a, b)
            for a, b in zip(values_a, values_b)
            if a is not None and b is not None
        ]
        if not pairs:
            return None
        return round(sum(1 for a, b in pairs if a == b) / len(pairs), 4)

    @staticmethod
    def _mean_or_none(a, b):
        values = [v for v in (a, b) if v is not None]
        return sum(values) / len(values) if values else None

    # ------------------------------------------------------------------
    # Datenzugriff
    # ------------------------------------------------------------------

    def _load_judge(self, path):
        """Judge-Urteile: Metriken aus der CSV, KM-Urteile aus dem Roh-JSON."""
        judged = {}
        for row in self._read_rows(path):
            km = {}
            raw = row.get("Judge_JSON", "")
            if raw.startswith("{"):
                judgment = json.loads(raw)
                km = {int(entry["id"]): bool(entry["covered"]) for entry in judgment["key_messages"]}
            judged[self._key(row)] = {
                "IA": self._to_float(row.get("IA")),
                "HR": self._to_float(row.get("HR")),
                "KMC": self._to_float(row.get("KMC")),
                "km": km,
            }
        if not judged:
            raise FileNotFoundError(f"Keine Judge-Urteile gefunden: {path}")
        return judged

    def _load_rating_sheet(self, path):
        """
        Menschliche Kodierung: Zählfelder -> Metriken (identische Formeln wie
        beim Judge), KM_i-Spalten -> binäre Urteile (1/ja/true vs. 0/nein/false).
        """
        ratings = {}
        for row in self._read_rows(path):
            korrekt = self._to_float(row.get("Claims_Korrekt"))
            intrinsisch = self._to_float(row.get("Claims_Intrinsisch"))
            extrinsisch = self._to_float(row.get("Claims_Extrinsisch"))
            if korrekt is None or intrinsisch is None or extrinsisch is None:
                continue  # unausgefüllte Zeile
            total = korrekt + intrinsisch + extrinsisch
            km = {}
            for column, value in row.items():
                if column.startswith("KM_") and value.strip() not in ("", "-"):
                    km[int(column[3:])] = self._parse_bool(value)
            km = {k: v for k, v in km.items() if v is not None}
            ratings[self._key(row)] = {
                "IA": round(korrekt / (korrekt + intrinsisch) * 100, 2) if korrekt + intrinsisch else None,
                "HR": round(extrinsisch / total * 100, 2) if total else None,
                "KMC": round(sum(km.values()) / len(km) * 100, 2) if km else None,
                "km": km,
            }
        if not ratings:
            raise ValueError(f"Kodierbogen ist leer oder unausgefüllt: {path}")
        return ratings

    def _try_load_rating_sheet(self, path):
        try:
            return self._load_rating_sheet(path)
        except (FileNotFoundError, ValueError):
            return None

    @staticmethod
    def _parse_bool(value):
        normalized = str(value).strip().lower()
        if normalized in ("1", "ja", "j", "true", "x", "yes"):
            return True
        if normalized in ("0", "nein", "n", "false"):
            return False
        return None

    @staticmethod
    def _read_rows(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=";"))

    @staticmethod
    def _key(row):
        return (row["Article_ID"], row["Method"], row["Model"], str(row["Seed"]))

    @staticmethod
    def _to_float(value):
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Ausgabe
    # ------------------------------------------------------------------

    def _print_report(self, results):
        print("\nÜbereinstimmung (Krippendorffs Alpha, Intervallskala):")
        for metric, row in results["metrics"].items():
            if row["degenerate"] and row["alpha_judge_gold"] is None:
                value = (
                    "Alpha nicht berechenbar (keine Varianz), "
                    f"exakte Übereinstimmung = {row['exact_agreement']}"
                )
            elif row["degenerate"]:
                value = (
                    f"Alpha degeneriert ({row['alpha_judge_gold']}, Prävalenzparadoxon, "
                    f"{row['deviating_units']} abweichende Einheit), "
                    f"exakte Übereinstimmung = {row['exact_agreement']}"
                )
            else:
                value = row["alpha_judge_gold"]
            parts = [f"  {metric}: Judge vs. Goldstandard = {value}"]
            if "alpha_rater1_rater2" in row:
                parts.append(f"Inter-Rater = {row['alpha_rater1_rater2']}")
            parts.append("GATE BESTANDEN" if row["gate"] else f"GATE VERFEHLT (< {self.ALPHA_GATE})")
            print(" | ".join(parts))
        if results["km_balanced_accuracy"]:
            print("Balanced Accuracy der Kernaussagen-Urteile:", results["km_balanced_accuracy"])
        all_passed = all(row["gate"] for row in results["metrics"].values())
        print(
            "\n=> Volllauf freigegeben." if all_passed
            else "\n=> Gate verfehlt: Judge-Prompt/Codebuch überarbeiten, dann Pilot wiederholen."
        )
