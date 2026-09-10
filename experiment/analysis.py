import csv
import os
import warnings

import numpy as np
from scipy import stats as scipy_stats


class ResultsAnalyzer:
    """
    Wertet die Judge-Urteile (judge_results.csv) gemaess dem vorab
    festgeschriebenen statistischen Verfahren aus (Abschnitt
    "Evaluierungsverfahren", Absatz "Statistisches Auswertungsverfahren"):

        RQ1 -- je Metrik (IA, HR, KMC), Modell und Optimierungsmethode wird
        der Methodeneffekt als Differenz zur Baseline "Original" gebildet,
        gepaart ueber Presseartikel x Seed (n = 15 Paare je Bedingung).
        Zweiseitiger Wilcoxon-Vorzeichen-Rang-Test (Signifikanzniveau 0,05),
        Holm-Korrektur ueber die sieben Methodenvergleiche je Metrik und
        Modell, Matched-Pairs-rangbiseriale Korrelation nach Kerby als
        Effektstaerke sowie Median und Interquartilsabstand der Differenzen.

        RQ2 -- Cross-Model Consistency: je Metrik die Spearman-
        Rangkorrelation der Effektvektoren beider Modelle ueber die
        n = 21 Methoden-Artikel-Bedingungen; die fuenf Seed-Wiederholungen
        je Bedingung werden zuvor gemittelt.

    Vorab fixierte Randfallregeln:
        - Nulldifferenzen werden nach Wilcoxons Originalverfahren von der
          Rangbildung ausgeschlossen; p-Werte exakt, bei Bindungen
          Normalapproximation (scipy method="auto").
        - Weist eine Bedingung ausschliesslich Nulldifferenzen auf, entfaellt
          der Test; berichtet wird deskriptiv "kein messbarer Effekt"
          (r = 0, p nicht definiert). Die Holm-Familie umfasst nur die
          tatsaechlich durchgefuehrten Tests.
        - Bei Rangbindungen wird die Spearman-Korrelation ueber die
          bindungskorrigierte Rang-Pearson-Form berechnet; ohne Varianz in
          einem Effektvektor ist sie nicht definiert.

    Die Klasse liest die Ergebnisdatei ausschliesslich lesend und rechnet
    lokal (keine Modellaufrufe). Vor der Auswertung wird die Vollstaendigkeit
    der 240er-Vollmatrix erzwungen.
    """

    METRICS = ("IA", "HR", "KMC")
    BASELINE = "Original"
    SIGNIFICANCE_LEVEL = 0.05

    MODEL_LABELS = {
        "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
        "gemini-3.5-flash": "Gemini 3.5 Flash",
    }
    # Schreibweise der Methoden identisch zum Fliesstext der Arbeit: Die
    # CamelCase-Schluessel der Ergebnisdateien werden nur fuer die Ausgabe in
    # Tabellen und Abbildungen aufgeloest, nicht in den Daten selbst.
    METHOD_LABELS = {
        "AuthoritativeTone": "Authoritative Tone",
        "ConclusionFirst": "Conclusion First",
        "FluencyOptimization": "Fluency Optimization",
        "LogicalStructure": "Logical Structure",
        "QuotationAddition": "Quotation Addition",
        "StatisticsAddition": "Statistics Addition",
    }
    # Farbrollen identisch zu tikz-bilder/tikz-palette.tex (Single Source of
    # Truth der Arbeit): Blau = Eingang/Fokus, Bernstein = Nebenpfad,
    # Gruen = validiertes Ergebnis, Anthrazit = neutral, Rot = Ausschluss.
    PALETTE = {
        "intro": "#37404A",   # Anthrazit  – neutral: Achsen, Nulllinien
        "theo": "#1F6FB2",    # Blau       – Eingang / Fokus
        "emp": "#2E8C6A",     # Gruen      – validiertes Ergebnis / Optimum
        "pra": "#C8772A",     # Bernstein  – Nebenpfad
        "err": "#C0392B",     # Rot        – Ausschluss
    }
    # Feste Zuordnung je Modell aus der Palette (Blau/Bernstein: in Farbton und
    # Helligkeit getrennt, daher auch bei Rot-Gruen-Schwaeche unterscheidbar).
    MODEL_COLORS = {
        "gpt-5.4-mini-2026-03-17": PALETTE["theo"],
        "gemini-3.5-flash": PALETTE["pra"],
    }
    FALLBACK_COLORS = (PALETTE["theo"], PALETTE["pra"])

    # Optimum je Metrik: IA/KMC werden maximiert, HR minimiert.
    METRIC_OPTIMUM = {"IA": 100.0, "HR": 0.0, "KMC": 100.0}
    METRIC_LABELS = {
        "IA": "Information Accuracy (%)",
        "HR": "Hallucination Rate (%)",
        "KMC": "Key Message Coverage (%)",
    }
    # Die RQ1-Abbildung zeigt nur die Metriken mit erkennbarer Streuung. Bei
    # der Information Accuracy sind nahezu alle gepaarten Differenzen null,
    # ein Boxplot waere dort eine leere Linie.
    RQ1_FIGURE_METRICS = ("HR", "KMC")

    def __init__(
        self,
        results_dir="../results",
        results_file="judge_results.csv",
        articles=("01", "02", "03"),
        methods=(
            "AuthoritativeTone", "ConclusionFirst", "FluencyOptimization",
            "JSON-LD", "LogicalStructure", "Original", "QuotationAddition",
            "StatisticsAddition",
        ),
        models=("gpt-5.4-mini-2026-03-17", "gemini-3.5-flash"),
        seeds=("42", "43", "44", "45", "46"),
    ):
        self.results_dir = results_dir
        self.results_file = results_file
        self.articles = tuple(articles)
        self.methods = tuple(methods)
        self.models = tuple(models)
        self.seeds = tuple(seeds)

    # ------------------------------------------------------------------
    # Oeffentliche Auswertung
    # ------------------------------------------------------------------

    def analyze(self):
        """Fuehrt die vollstaendige RQ1/RQ2-Auswertung durch."""
        data = self._load()
        results = {
            "n_units": len(data),
            "rq1": [], "rq2": [], "deskriptiv": [], "explorativ": [],
            "rq1_diffs": {}, "rq2_deltas": {},
            "explorativ_diffs": {}, "explorativ_km": {},
            "werte": data,
        }
        self._analyze_rq1(data, results)
        self._analyze_rq2(data, results)
        self._analyze_descriptive(data, results)
        self._analyze_exploratory(data, results)
        return results

    def report(self):
        """Auswertung mit formatierter Konsolenausgabe; gibt die Ergebnisse zurueck."""
        results = self.analyze()
        self._print_report(results)
        return results

    # ------------------------------------------------------------------
    # RQ1: Methodeneffekte gegen die Baseline
    # ------------------------------------------------------------------

    def _analyze_rq1(self, data, results):
        pairs = [(a, s) for a in self.articles for s in self.seeds]
        test_methods = [m for m in self.methods if m != self.BASELINE]

        for metric in self.METRICS:
            for model in self.models:
                family = []
                for method in test_methods:
                    diffs = [
                        data[(a, method, model, s)][metric]
                        - data[(a, self.BASELINE, model, s)][metric]
                        for a, s in pairs
                    ]
                    nonzero = [d for d in diffs if d != 0]
                    p_value = self._wilcoxon(diffs)
                    row = {
                        "metric": metric,
                        "model": model,
                        "method": method,
                        "n_pairs": len(diffs),
                        "n_eff": len(nonzero),
                        "median": float(np.median(diffs)),
                        "q25": float(np.percentile(diffs, 25)),
                        "q75": float(np.percentile(diffs, 75)),
                        "p_value": p_value,
                        "r_rb": self._rank_biserial(diffs),
                        "degenerate": len(nonzero) == 0,
                    }
                    results["rq1_diffs"][(metric, model, method)] = diffs
                    family.append(row)

                adjusted = self._holm([row["p_value"] for row in family])
                for row, p_holm in zip(family, adjusted):
                    row["p_holm"] = p_holm
                    row["significant"] = (
                        p_holm is not None and p_holm < self.SIGNIFICANCE_LEVEL
                    )
                results["rq1"].extend(family)

    # ------------------------------------------------------------------
    # RQ2: Cross-Model Consistency
    # ------------------------------------------------------------------

    def _analyze_rq2(self, data, results):
        if len(self.models) != 2:
            return
        model_a, model_b = self.models
        test_methods = [m for m in self.methods if m != self.BASELINE]
        labels = [(m, a) for m in test_methods for a in self.articles]

        def condition_mean(model, method, article):
            values = [data[(article, method, model, s)] for s in self.seeds]
            return {
                metric: float(np.mean([v[metric] for v in values]))
                for metric in self.METRICS
            }

        means = {
            (model, method, article): condition_mean(model, method, article)
            for model in self.models
            for method in self.methods
            for article in self.articles
        }

        for metric in self.METRICS:
            deltas = {}
            for model in self.models:
                deltas[model] = [
                    means[(model, method, article)][metric]
                    - means[(model, self.BASELINE, article)][metric]
                    for method, article in labels
                ]
            rho, p_value = self._spearman(deltas[model_a], deltas[model_b])
            results["rq2"].append({
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "rho": rho,
                "p_value": p_value,
                "n": len(labels),
            })
            results["rq2_deltas"][metric] = {
                "labels": labels,
                model_a: deltas[model_a],
                model_b: deltas[model_b],
            }

    # ------------------------------------------------------------------
    # Deskriptivteil
    # ------------------------------------------------------------------

    def _analyze_descriptive(self, data, results):
        for model in self.models:
            for method in self.methods:
                values = [
                    data[(a, method, model, s)]
                    for a in self.articles for s in self.seeds
                ]
                row = {"model": model, "method": method, "n": len(values)}
                for metric in self.METRICS:
                    series = [v[metric] for v in values]
                    row[f"{metric}_mean"] = float(np.mean(series))
                    row[f"{metric}_sd"] = float(np.std(series, ddof=1))
                results["deskriptiv"].append(row)

    # ------------------------------------------------------------------
    # Explorative Zusatzanalyse: Umfang der Wiedergabe
    # ------------------------------------------------------------------

    def _analyze_exploratory(self, data, results):
        """
        Nicht vorab festgelegte Zusatzanalyse zum Antwortumfang (Anzahl der
        vom Evaluator identifizierten atomaren Aussagen). Unkorrigiert, rein
        hypothesengenerierend; die Ergebnisse tragen keine konfirmatorische
        Aussage und gehen nicht in die Holm-Familien von RQ1 ein.
        """
        pairs = [(a, s) for a in self.articles for s in self.seeds]
        if any(data[(a, m, mo, s)]["claims"] is None
               for a in self.articles for m in self.methods
               for mo in self.models for s in self.seeds):
            return  # Zaehlspalten nicht vorhanden -> Analyse entfaellt

        for method in (m for m in self.methods if m != self.BASELINE):
            for model in self.models:
                base = [data[(a, self.BASELINE, model, s)]["claims"] for a, s in pairs]
                var = [data[(a, method, model, s)]["claims"] for a, s in pairs]
                diffs = [v - b for v, b in zip(var, base)]
                mean_base, mean_var = float(np.mean(base)), float(np.mean(var))
                results["explorativ"].append({
                    "model": model, "method": method,
                    "median": float(np.median(diffs)),
                    "mean_base": mean_base, "mean_var": mean_var,
                    "rel_change": (mean_var / mean_base - 1.0) * 100.0,
                    "p_value": self._wilcoxon(diffs),
                    "r_rb": self._rank_biserial(diffs),
                    "n_eff": sum(1 for d in diffs if d != 0),
                })
                results["explorativ_diffs"][(model, method)] = diffs

        # Dissoziation: Umfang vs. abgedeckte Kernaussagen unter der Methode
        # mit dem staerksten Umfangseffekt (im Datensatz: Conclusion First).
        for model in self.models:
            for method in (m for m in self.methods if m != self.BASELINE):
                km_base = [data[(a, self.BASELINE, model, s)]["km_covered"]
                           for a, s in pairs]
                km_var = [data[(a, method, model, s)]["km_covered"] for a, s in pairs]
                results["explorativ_km"][(model, method)] = {
                    "mean_base": float(np.mean(km_base)),
                    "mean_var": float(np.mean(km_var)),
                    "rel_change": (float(np.mean(km_var)) / float(np.mean(km_base))
                                   - 1.0) * 100.0,
                    "sum_diff": int(sum(v - b for v, b in zip(km_var, km_base))),
                }

    # ------------------------------------------------------------------
    # Statistische Kennzahlen
    # ------------------------------------------------------------------

    @staticmethod
    def _wilcoxon(diffs):
        """
        Zweiseitiger Wilcoxon-Vorzeichen-Rang-Test. Nulldifferenzen werden
        vorab ausgeschlossen (Wilcoxon-Originalverfahren), sodass der Test
        ohne Bindungen exakt bleibt; None, wenn keine Differenz verbleibt.
        """
        nonzero = [d for d in diffs if d != 0]
        if not nonzero:
            return None
        result = scipy_stats.wilcoxon(
            nonzero, zero_method="wilcox", method="auto", alternative="two-sided"
        )
        return float(result.pvalue)

    @staticmethod
    def _rank_biserial(diffs):
        """
        Matched-Pairs-rangbiseriale Korrelation nach Kerby:
        r = (R+ - R-) / (R+ + R-) auf den Nicht-Null-Differenzen;
        0.0 bei ausschliesslich Nulldifferenzen (kein messbarer Effekt).
        """
        nonzero = [d for d in diffs if d != 0]
        if not nonzero:
            return 0.0
        ranks = scipy_stats.rankdata([abs(d) for d in nonzero])
        r_plus = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
        r_minus = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))
        return (r_plus - r_minus) / (r_plus + r_minus)

    @staticmethod
    def _holm(pvalues):
        """
        Holm-Korrektur (Step-down). None-Eintraege (degenerierte Bedingungen
        ohne Test) bleiben None und zaehlen nicht zur Familiengroesse.
        """
        indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
        adjusted = [None] * len(pvalues)
        m = len(indexed)
        previous = 0.0
        for rank, (i, p) in enumerate(sorted(indexed, key=lambda item: item[1])):
            value = min(1.0, max(previous, (m - rank) * p))
            value = round(value, 10)  # entfernt Gleitkomma-Artefakte
            adjusted[i] = value
            previous = value
        return adjusted

    @staticmethod
    def _spearman(values_a, values_b):
        """Bindungskorrigierte Spearman-Korrelation; (None, None) ohne Varianz."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scipy_stats.spearmanr(values_a, values_b)
        rho = float(result.statistic)
        if np.isnan(rho):
            return None, None
        return rho, float(result.pvalue)

    # ------------------------------------------------------------------
    # Datenzugriff und Vollstaendigkeits-Gate
    # ------------------------------------------------------------------

    def _load(self):
        path = os.path.join(self.results_dir, self.results_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter=";"))

        data = {}
        for row in rows:
            key = (row["Article_ID"], row["Method"], row["Model"], str(row["Seed"]))
            if key in data:
                raise ValueError(f"Doppelter Bewertungsschluessel in {path}: {key}")
            metrics = {}
            for metric in self.METRICS:
                value = self._to_float(row.get(metric))
                if value is None:
                    raise ValueError(f"Fehlender {metric}-Wert fuer {key} in {path}")
                metrics[metric] = value
            # Zaehlgroessen der explorativen Zusatzanalyse: optional, da sie
            # keine der konfirmatorischen Kennzahlen speisen. Fehlen sie,
            # entfaellt die explorative Auswertung ersatzlos.
            for field, name in (("Claims_Gesamt", "claims"),
                                ("KM_Covered", "km_covered")):
                metrics[name] = self._to_float(row.get(field))
            data[key] = metrics

        expected = {
            (a, m, mo, s)
            for a in self.articles for m in self.methods
            for mo in self.models for s in self.seeds
        }
        unexpected = sorted(set(data) - expected)
        if unexpected:
            raise ValueError(
                f"Unerwartete Bewertungsschluessel ausserhalb der Vollmatrix: {unexpected}"
            )
        missing = sorted(expected - set(data))
        if missing:
            listing = "\n".join(f"  {key}" for key in missing)
            raise ValueError(
                f"Auswertung abgebrochen: es fehlen {len(missing)} von "
                f"{len(expected)} Urteilen in {path}:\n{listing}"
            )
        return data

    @staticmethod
    def _to_float(value):
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Exporte: CSV, LaTeX-Tabellen, Abbildungen
    # ------------------------------------------------------------------

    RQ1_COLUMNS = (
        "metric", "model", "method", "n_pairs", "n_eff", "median", "q25", "q75",
        "p_value", "p_holm", "significant", "r_rb", "degenerate",
    )
    RQ2_COLUMNS = ("metric", "model_a", "model_b", "rho", "p_value", "n")

    def export_csv(self, results):
        """Schreibt analysis_rq1/rq2/deskriptiv.csv in das Ergebnisverzeichnis."""
        deskriptiv_columns = ["model", "method", "n"] + [
            f"{metric}_{stat}" for metric in self.METRICS for stat in ("mean", "sd")
        ]
        jobs = (
            ("analysis_rq1.csv", self.RQ1_COLUMNS, results["rq1"]),
            ("analysis_rq2.csv", self.RQ2_COLUMNS, results["rq2"]),
            ("analysis_deskriptiv.csv", deskriptiv_columns, results["deskriptiv"]),
        )
        paths = []
        for name, columns, rows in jobs:
            path = os.path.join(self.results_dir, name)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(columns), delimiter=";")
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        c: self._csv_value(row.get(c)) for c in columns
                    })
            paths.append(path)
        return paths

    @staticmethod
    def _csv_value(value):
        if isinstance(value, float):
            return round(value, 6)
        return value

    def export_latex(self, results, tables_dir):
        """
        Schreibt \\input-faehige booktabs-Fragmente (nur tabular-Umgebung,
        deutsche Zahlformatierung) fuer die Ergebniskapitel.
        """
        os.makedirs(tables_dir, exist_ok=True)
        paths = []
        path = os.path.join(tables_dir, "rq1_methodeneffekte.tex")
        self._write_file(path, self._latex_rq1(results))
        paths.append(path)
        path = os.path.join(tables_dir, "rq2_konsistenz.tex")
        self._write_file(path, self._latex_rq2(results))
        paths.append(path)
        path = os.path.join(tables_dir, "deskriptiv.tex")
        self._write_file(path, self._latex_descriptive(results))
        paths.append(path)
        return paths

    def _latex_rq1(self, results):
        """
        Alle drei Metriken in einer Tabelle: gleiche Kennzahlen, gleiche Zeilen,
        daher ein Kopf statt drei. Die Metriken trennen kursive Gruppenzeilen.
        """
        test_methods = [m for m in self.methods if m != self.BASELINE]
        span = 1 + 5 * len(self.models)
        lines = [
            "\\begin{tabular}{l" + "rrrrr" * len(self.models) + "}",
            "  \\toprule",
        ]
        header = ["  "]
        for model in self.models:
            header.append(
                f"& \\multicolumn{{5}}{{c}}{{{self._model_label(model)}}} "
            )
        lines.append("".join(header) + "\\\\")
        ranges = [
            f"  \\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}"
            for i in range(len(self.models))
        ]
        lines.append("".join(ranges))
        lines.append(
            "  Methode "
            + "& Median~$\\Delta$ & IQA & $p_{\\text{Holm}}$ & $r$ & $n_{\\text{eff}}$ "
            * len(self.models)
            + "\\\\"
        )
        lines.append("  \\midrule")
        for index, metric in enumerate(self.METRICS):
            if index:
                lines.append("  \\addlinespace[4pt]")
            label = self.METRIC_LABELS.get(metric, metric).replace(" (%)", "")
            lines.append(
                f"  \\multicolumn{{{span}}}{{@{{}}l}}{{\\bfseries "
                f"{self._latex_escape(label)}}} \\\\[2pt]"
            )
            for method in test_methods:
                cells = [f"  {self._latex_escape(self._method_label(method))} "]
                for model in self.models:
                    row = next(
                        r for r in results["rq1"]
                        if r["metric"] == metric and r["model"] == model
                        and r["method"] == method
                    )
                    p_text = self._fmt_p(row["p_holm"])
                    if row["significant"]:
                        p_text = f"\\textbf{{{p_text}}}"
                    cells.append(
                        f"& {self._fmt_tex(row['median'], 2)} "
                        f"& {self._fmt_tex(row['q75'] - row['q25'], 2)} "
                        f"& {p_text} "
                        f"& {self._fmt_tex(row['r_rb'], 2)} "
                        f"& {row['n_eff']} "
                    )
                lines.append("".join(cells) + "\\\\")
        lines += ["  \\bottomrule", "\\end{tabular}", ""]
        return "\n".join(lines)

    def _latex_rq2(self, results):
        lines = [
            "\\begin{tabular}{lrrr}",
            "  \\toprule",
            "  Metrik & $\\rho$ & $p$ & $n$ \\\\",
            "  \\midrule",
        ]
        for row in results["rq2"]:
            rho = self._fmt_tex(row["rho"], 2) if row["rho"] is not None else "--"
            lines.append(
                f"  {row['metric']} & {rho} & {self._fmt_p(row['p_value'])} "
                f"& {row['n']} \\\\"
            )
        lines += ["  \\bottomrule", "\\end{tabular}", ""]
        return "\n".join(lines)

    def _latex_descriptive(self, results):
        lines = [
            "\\begin{tabular}{l" + "rrr" * len(self.models) + "}",
            "  \\toprule",
            "  "
            + "".join(
                f"& \\multicolumn{{3}}{{c}}{{{self._model_label(model)}}} "
                for model in self.models
            )
            + "\\\\",
            "".join(
                f"  \\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                for i in range(len(self.models))
            ),
            "  Variante "
            + "& IA & HR & KMC " * len(self.models)
            + "\\\\",
            "  \\midrule",
        ]
        for method in self.methods:
            cells = [f"  {self._latex_escape(self._method_label(method))} "]
            for model in self.models:
                row = next(
                    r for r in results["deskriptiv"]
                    if r["model"] == model and r["method"] == method
                )
                for metric in self.METRICS:
                    cells.append(
                        f"& {self._fmt_tex(row[f'{metric}_mean'], 1)} "
                        f"({self._fmt_tex(row[f'{metric}_sd'], 1)}) "
                    )
            lines.append("".join(cells) + "\\\\")
        lines += ["  \\bottomrule", "\\end{tabular}", ""]
        return "\n".join(lines)

    # Schriftsetzung identisch zu thesis.tex: pdflatex + T1 + Latin Modern.
    # Die Abbildungstexte werden dadurch von derselben Engine gesetzt wie das
    # Dokument, statt nur eine aehnliche Schrift zu imitieren.
    TEX_SYSTEM = "pdflatex"
    TEX_PREAMBLE = "\n".join((
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
    ))

    def _configure_text_rendering(self, matplotlib):
        """
        Schaltet die Textausgabe auf echtes LaTeX (pgf-Backend). Ist keine
        LaTeX-Installation erreichbar, wird auf die matplotlib-Standardschrift
        zurueckgefallen, damit die Auswertung auch ohne TeX lauffaehig bleibt.
        Setzt self._usetex und gibt den erreichten Modus zurueck.
        """
        import shutil

        if shutil.which(self.TEX_SYSTEM) is None:
            warnings.warn(
                f"{self.TEX_SYSTEM} nicht gefunden: Abbildungen werden mit der "
                "matplotlib-Standardschrift statt mit Latin Modern gesetzt.",
                RuntimeWarning, stacklevel=2,
            )
            matplotlib.use("Agg")
            self._usetex = False
            return False

        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": self.TEX_SYSTEM,
            "pgf.preamble": self.TEX_PREAMBLE,
            "pgf.rcfonts": False,   # Schrift kommt aus der Praeambel, nicht aus rcParams
            "text.usetex": True,
            "font.family": "serif",
        })
        self._usetex = True
        return True

    def _tex(self, text):
        """Maskiert das Prozentzeichen, das LaTeX sonst als Kommentar liest."""
        return text.replace("%", r"\%") if getattr(self, "_usetex", False) else text

    def export_figures(self, results, images_dir):
        """
        Vektor-PDFs fuer die Ergebniskapitel: je Metrik Boxplots der gepaarten
        Differenzen (RQ1) sowie das Streudiagramm der Effektvektoren (RQ2).
        Die Texte werden nach Moeglichkeit von LaTeX gesetzt (vgl.
        _configure_text_rendering), damit die Schrift der des Dokuments
        entspricht.
        """
        import matplotlib
        self._configure_text_rendering(matplotlib)
        import matplotlib.pyplot as plt

        os.makedirs(images_dir, exist_ok=True)
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
            "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
        })
        paths = []
        path = os.path.join(images_dir, "rq1_effekte.pdf")
        self._figure_rq1(plt, results, path)
        paths.append(path)
        path = os.path.join(images_dir, "rq2_konsistenz.pdf")
        self._figure_rq2(plt, results, path)
        paths.append(path)

        path = os.path.join(images_dir, "deckensaettigung.pdf")
        self._figure_saturation(plt, results, path)
        paths.append(path)
        if results.get("explorativ"):
            path = os.path.join(images_dir, "explorativ_umfang.pdf")
            self._figure_exploratory(plt, results, path)
            paths.append(path)
        return paths

    def _figure_saturation(self, plt, results, path, highlight_seed=20260805):
        """
        Absolute Metrikniveaus aller Bewertungseinheiten je Metrik und Modell.
        Macht die Deckensaettigung sichtbar, die der Nullbefund voraussetzt:
        die Punktmasse liegt auf dem Optimum, Abweichungen sind Einzelfaelle.
        Der Jitter ist ueber einen festen Seed reproduzierbar.
        """
        data = results["werte"]
        rng = np.random.default_rng(highlight_seed)
        fig, axes = plt.subplots(
            len(self.METRICS), 1, figsize=(6.3, 4.6), constrained_layout=True
        )
        lanes = list(enumerate(reversed(self.models)))  # erstes Modell oben

        for ax, metric in zip(axes, self.METRICS):
            optimum = self.METRIC_OPTIMUM[metric]
            allvals = [
                data[(a, m, mo, s)][metric]
                for a in self.articles for m in self.methods
                for mo in self.models for s in self.seeds
            ]
            lo, hi = min(allvals + [optimum]), max(allvals + [optimum])
            span = (hi - lo) or 1.0
            at_max_side = optimum >= hi

            for lane, model in lanes:
                values = np.array([
                    data[(a, m, model, s)][metric]
                    for a in self.articles for m in self.methods for s in self.seeds
                ])
                jitter = rng.uniform(-0.30, 0.30, size=len(values))
                ax.scatter(
                    values, lane + jitter, s=9, alpha=0.45, linewidths=0,
                    color=self._model_color(model, self.models.index(model)),
                    zorder=3,
                )
                n_opt = int(np.sum(values == optimum))
                ax.annotate(
                    f"{n_opt}/{len(values)} am Optimum",
                    xy=(optimum, lane),
                    xytext=(optimum + (0.035 if at_max_side else -0.035) * span, lane),
                    ha="left" if at_max_side else "right", va="center",
                    fontsize=7.5, color=self.PALETTE["intro"], zorder=4,
                )

            ax.axvline(optimum, color=self.PALETTE["emp"], linewidth=1.1,
                       linestyle="--", zorder=2)
            pad, room = 0.06 * span, 0.40 * span
            ax.set_xlim(*(lo - pad, hi + room) if at_max_side
                        else (lo - room, hi + pad))
            # Der Randbereich traegt ausschliesslich die Annotation. Die Ticks
            # werden auf den moeglichen Wertebereich begrenzt, damit die Achse
            # keine unmoeglichen Werte (IA/KMC > 100 %, HR < 0 %) suggeriert.
            from matplotlib.ticker import MaxNLocator
            locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
            ticks = [t for t in locator.tick_values(lo, hi)
                     if lo - 1e-9 <= t <= hi + 1e-9]
            if all(abs(t - optimum) > 1e-9 for t in ticks):
                ticks = sorted(ticks + [optimum])  # Optimum immer beschriften
            ax.set_xticks(ticks)
            ax.set_ylim(-0.62, len(self.models) - 0.38)
            ax.set_yticks([lane for lane, _ in lanes])
            ax.set_yticklabels([self._model_label(m) for _, m in lanes], fontsize=8)
            ax.set_xlabel(self._tex(self.METRIC_LABELS.get(metric, metric)),
                          fontsize=8.5)
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)

        handles = [
            plt.Line2D([], [], color=self._model_color(m, i), marker="o",
                       linestyle="none", markersize=4, label=self._model_label(m))
            for i, m in enumerate(self.models)
        ] + [plt.Line2D([], [], color=self.PALETTE["emp"], linewidth=1.1,
                        linestyle="--", label="Optimum der Metrik")]
        fig.legend(handles=handles, frameon=False, fontsize=8,
                   loc="outside lower center", ncol=3)
        fig.savefig(path, metadata={"CreationDate": None})
        plt.close(fig)

    def _figure_exploratory(self, plt, results, path, focus="ConclusionFirst"):
        """
        Explorative Zusatzanalyse zum Antwortumfang (nicht vorab festgelegt).
        Links: Median-Effekt aller Methoden auf die Anzahl atomarer Aussagen.
        Rechts: Dissoziation unter der Fokusmethode - der Umfang sinkt,
        die abgedeckten Kernaussagen bleiben praktisch unveraendert.
        """
        methods = [m for m in self.methods if m != self.BASELINE]
        rows = {(r["model"], r["method"]): r for r in results["explorativ"]}
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(6.3, 3.3), width_ratios=(1.45, 1.15),
            constrained_layout=True,
        )

        # (a) Median-Effekt je Methode und Modell
        ypos = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
        if focus in ypos:
            ax_a.axhspan(ypos[focus] - 0.45, ypos[focus] + 0.45,
                         color=self.PALETTE["intro"], alpha=0.07, zorder=0)
        ax_a.axvline(0, color=self.PALETTE["intro"], linewidth=0.9, zorder=1)
        for idx, model in enumerate(self.models):
            offset = (idx - (len(self.models) - 1) / 2) * 0.30
            ax_a.scatter(
                [rows[(model, m)]["median"] for m in methods],
                [ypos[m] + offset for m in methods],
                s=26, color=self._model_color(model, idx),
                edgecolors="white", linewidths=0.5, zorder=3,
            )
            if focus in ypos:
                row = rows[(model, focus)]
                # Rechts vom Marker: die Fokusmethode hat die staerksten
                # negativen Mediane, links davon ist kein Platz.
                ax_a.annotate(
                    f"p = {self._fmt(row['p_value'], 3)}",
                    xy=(row["median"], ypos[focus] + offset),
                    xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7,
                    color=self._model_color(model, idx), zorder=4,
                )
        from matplotlib.ticker import MaxNLocator
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))  # Aussagen sind Zaehlwerte
        ax_a.margins(x=0.10)
        ax_a.set_yticks(list(ypos.values()))
        ax_a.set_yticklabels(
            [self._method_label(m) for m in ypos], fontsize=8
        )
        ax_a.set_ylim(-0.6, len(methods) - 0.4)
        ax_a.set_xlabel("Median $\\Delta$ atomare Aussagen ggü. Original",
                        fontsize=8.5)
        ax_a.tick_params(axis="x", labelsize=8)
        ax_a.xaxis.grid(True, color="#dddddd", linewidth=0.6)
        ax_a.set_axisbelow(True)
        # Zweizeilig wie Panel (b), damit beide Plotflaechen buendig abschliessen.
        ax_a.set_title("(a) Effekt auf den\nAntwortumfang", fontsize=9, loc="left")

        # (b) Dissoziation Umfang vs. Kernaussagenabdeckung
        groups = ("atomare\nAussagen", "abgedeckte\nKernaussagen")
        width = 0.30  # schmaler als der Gruppenabstand: Platz fuer die Wertelabels
        ax_b.axhline(0, color=self.PALETTE["intro"], linewidth=0.9, zorder=1)
        spread = [
            v for model in self.models
            for v in (rows[(model, focus)]["rel_change"],
                      results["explorativ_km"][(model, focus)]["rel_change"])
        ]
        margin = 0.24 * (max(spread) - min(spread)) or 1.0
        ax_b.set_ylim(min(spread) - margin, max(spread) + margin)
        for idx, model in enumerate(self.models):
            offset = (idx - (len(self.models) - 1) / 2) * width
            values = (
                rows[(model, focus)]["rel_change"],
                results["explorativ_km"][(model, focus)]["rel_change"],
            )
            ax_b.bar(
                [i + offset for i in range(len(groups))], values, width=width * 0.92,
                color=self._model_color(model, idx), linewidth=0, zorder=3,
            )
            for i, v in enumerate(values):
                ax_b.annotate(
                    self._tex(f"{v:+.1f}".replace(".", ",") + " %"),
                    xy=(i + offset, v), xytext=(0, -11 if v < 0 else 3),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=self.PALETTE["intro"], zorder=4,
                )
        ax_b.set_xticks(range(len(groups)))
        ax_b.set_xticklabels(groups, fontsize=8)
        ax_b.set_ylabel(self._tex("Veränderung ggü. Original (%)"), fontsize=8.5)
        ax_b.tick_params(axis="y", labelsize=8)
        ax_b.yaxis.grid(True, color="#dddddd", linewidth=0.6)
        ax_b.set_axisbelow(True)
        ax_b.set_title(f"(b) Dissoziation unter\n{self._method_label(focus)}",
                       fontsize=9, loc="left")

        handles = [
            plt.Line2D([], [], color=self._model_color(m, i), marker="o",
                       linestyle="none", markersize=4, label=self._model_label(m))
            for i, m in enumerate(self.models)
        ]
        fig.legend(handles=handles, frameon=False, fontsize=8,
                   loc="outside lower center", ncol=2)
        fig.savefig(path, metadata={"CreationDate": None})
        plt.close(fig)

    def _figure_rq1(self, plt, results, path):
        """
        Gepaarte Differenzen je Optimierungsmethode, ein Feld je Metrik mit
        gemeinsamer x-Achse. Die Metriken teilen Achsenbeschriftung und
        Legende, weil der Fliesstext sie gemeinsam auswertet; das haelt
        zugleich die Float-Masse des Ergebnisabschnitts so klein, dass die
        Abbildung im eigenen Abschnitt gesetzt werden kann.
        """
        test_methods = [m for m in self.methods if m != self.BASELINE]
        metrics = self.RQ1_FIGURE_METRICS
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(6.3, 4.0), sharex=True,
            constrained_layout=True,
        )
        n_models = len(self.models)
        width = 0.7 / n_models
        for panel, (ax, metric) in enumerate(zip(axes, metrics)):
            for idx, model in enumerate(self.models):
                color = self._model_color(model, idx)
                offset = (idx - (n_models - 1) / 2) * width
                positions = [i + offset for i in range(len(test_methods))]
                values = [
                    results["rq1_diffs"][(metric, model, method)]
                    for method in test_methods
                ]
                ax.boxplot(
                    values, positions=positions, widths=width * 0.85,
                    boxprops={"color": color, "linewidth": 1.2},
                    whiskerprops={"color": color, "linewidth": 1.0},
                    capprops={"color": color, "linewidth": 1.0},
                    medianprops={"color": color, "linewidth": 1.6},
                    flierprops={
                        "marker": "o", "markersize": 3,
                        "markerfacecolor": color, "markeredgecolor": "none",
                    },
                )
            ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
            label = self.METRIC_LABELS.get(metric, metric).replace(" (%)", "")
            ax.set_title(f"({chr(97 + panel)}) {label}", fontsize=9, loc="left")
            ax.set_ylabel(f"$\\Delta$ {metric} (Prozentpunkte)")
            ax.yaxis.grid(True, color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
        axes[-1].set_xticks(range(len(test_methods)))
        axes[-1].set_xticklabels(
            [self._method_label(m) for m in test_methods], rotation=25, ha="right"
        )
        handles = [
            plt.Line2D([], [], color=self._model_color(model, idx), linewidth=2,
                       label=self._model_label(model))
            for idx, model in enumerate(self.models)
        ]
        # Legende ausserhalb der Datenflaeche: bei "best" wandert sie je nach
        # Figurhoehe in die Punktwolke und verdeckt Ausreisser.
        fig.legend(
            handles=handles, loc="outside upper right", ncol=len(handles),
            frameon=False, fontsize=8,
        )
        fig.savefig(path, metadata={"CreationDate": None})
        plt.close(fig)

    def _figure_rq2(self, plt, results, path):
        model_a, model_b = self.models
        fig, axes = plt.subplots(
            1, len(self.METRICS), figsize=(6.3, 2.5), constrained_layout=True
        )
        for ax, metric in zip(axes, self.METRICS):
            deltas = results["rq2_deltas"][metric]
            row = next(r for r in results["rq2"] if r["metric"] == metric)
            ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
            ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
            ax.scatter(
                deltas[model_a], deltas[model_b], s=20,
                color=self._model_color(model_a, 0),
                edgecolors="white", linewidths=0.5, zorder=2,
            )
            if row["rho"] is not None:
                # Mathematisches Minus statt Bindestrich; als Mathe-Modus
                # notiert, damit es unter LaTeX wie unter mathtext traegt.
                rho = self._fmt(row["rho"], 2).replace("-", "$-$")
            else:
                # Nicht "--": LaTeX ligiert das zu einem Halbgeviertstrich,
                # der neben "rho =" wie ein Minuszeichen zu lesen waere.
                rho = "n. def."
            ax.set_title(f"{metric}  ($\\rho$ = {rho})")
            ax.set_xlabel(f"$\\Delta$ {self._model_label(model_a)}")
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel(f"$\\Delta$ {self._model_label(model_b)}")
        fig.savefig(path, metadata={"CreationDate": None})
        plt.close(fig)

    # ------------------------------------------------------------------
    # Formatierung und Ausgabe
    # ------------------------------------------------------------------

    def _model_label(self, model):
        return self.MODEL_LABELS.get(model, model)

    def _method_label(self, method):
        return self.METHOD_LABELS.get(method, method)

    def _model_color(self, model, index):
        return self.MODEL_COLORS.get(
            model, self.FALLBACK_COLORS[index % len(self.FALLBACK_COLORS)]
        )

    @staticmethod
    def _fmt(value, decimals):
        """Deutsche Zahlformatierung (Dezimalkomma) fuer LaTeX/Berichte."""
        return f"{value:.{decimals}f}".replace(".", ",")

    @classmethod
    def _fmt_tex(cls, value, decimals):
        """Wie _fmt, aber mit typographischem Minuszeichen fuer LaTeX-Tabellen."""
        text = cls._fmt(value, decimals)
        if text.startswith("-"):
            return "$-$" + text[1:]
        return text

    @classmethod
    def _fmt_p(cls, p_value, latex=True):
        if p_value is None:
            return "--"
        if p_value < 0.001:
            return "$<$0,001" if latex else "<0,001"
        return cls._fmt(p_value, 3)

    @staticmethod
    def _write_file(path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _latex_escape(text):
        return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    def _print_report(self, results):
        print(f"Bewertungseinheiten: {results['n_units']}")
        print("\nRQ1 -- Methodeneffekte gegen die Baseline (Wilcoxon, Holm-korrigiert):")
        for metric in self.METRICS:
            print(f"\n  {metric}:")
            for model in self.models:
                print(f"    {self._model_label(model)}:")
                for row in results["rq1"]:
                    if row["metric"] != metric or row["model"] != model:
                        continue
                    if row["degenerate"]:
                        detail = "alle Differenzen 0 -> kein messbarer Effekt"
                    else:
                        marker = " *" if row["significant"] else ""
                        detail = (
                            f"Median dif = {self._fmt(row['median'], 2)}, "
                            f"p_Holm = {self._fmt_p(row['p_holm'], latex=False)}{marker}, "
                            f"r = {self._fmt(row['r_rb'], 2)}, "
                            f"n_eff = {row['n_eff']}"
                        )
                    print(f"      {row['method']:<20s} {detail}")
        print("\nRQ2 -- Cross-Model Consistency (Spearman, Seeds gemittelt):")
        for row in results["rq2"]:
            rho = self._fmt(row["rho"], 4) if row["rho"] is not None else "nicht definiert (keine Varianz)"
            p_text = self._fmt_p(row["p_value"], latex=False)
            print(f"  {row['metric']}: rho = {rho} (p = {p_text}, n = {row['n']})")
        significant = [r for r in results["rq1"] if r["significant"]]
        print(f"\n=> {len(significant)} von {len(results['rq1'])} RQ1-Vergleichen "
              f"signifikant (Holm-korrigiert, Niveau 0,05).")
