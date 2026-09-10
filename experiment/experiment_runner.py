import csv
import os
from datetime import datetime

import pandas as pd


class ExperimentRunner:
    """
    Steuert den Versuchsablauf. Ein Durchlauf (Replikation) entspricht genau
    einem Seed aus dem kanonischen Seed-Satz und umfasst alle Artikelvarianten
    und Modelle.

    Durchläufe sind einzeln startbar (run_replication) und idempotent:
    Vollständige Durchläufe werden übersprungen, unvollständige nur aufgefüllt,
    sodass kein API-Aufruf doppelt bezahlt wird. Jede Antwort wird unmittelbar
    nach Erhalt gespeichert; je Durchlauf entsteht eine separate CSV-Datei nach
    dem Datenschema aus dem Anhang der Thesis (Tabelle "csv-schema").
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Du bist ein präziser Analyst. Antworte kurz und faktenbasiert."
    )
    DEFAULT_USER_PROMPT = "Artikel:\n{article}\n\nAnalysiere den Artikel."

    # Kanonischer Seed-Satz (n = 5 Replikationen); Durchlauf N nutzt Seed[N-1]
    # für jede Variante und beide Modelle identisch (Abschnitt 4.3.1 der Thesis).
    DEFAULT_SEEDS = [42, 43, 44, 45, 46]

    # Spaltenreihenfolge der Ergebnisdatei gemäß Anhang der Thesis.
    CSV_COLUMNS = [
        "Timestamp",
        "Article_ID",
        "Article_Type",
        "Method",
        "Seed",
        "Model",
        "Model_Version",
        "System_Prompt",
        "User_Prompt",
        "Answer",
        "Prompt_Tokens",
        "Completion_Tokens",
        "Total_Tokens",
    ]

    def __init__(
        self,
        loader,
        connector_list,
        prompt_config=None,
        prompt_configs=None,
        article_prompt_configs=None,
        article_folder="processed",
        article_types=None,
        seeds=None,
        prices=None,
    ):
        self.loader = loader
        self.connectors = connector_list  # Liste von LLMConnector-Instanzen
        self.article_folder = article_folder
        self.prompt_config = self._normalize_prompt_config(prompt_config)
        if article_prompt_configs is not None and prompt_configs is not None:
            raise ValueError(
                "Nutze entweder prompt_configs oder article_prompt_configs, nicht beides."
            )
        self.article_prompt_configs = self._normalize_prompt_config_mapping(
            article_prompt_configs if article_prompt_configs is not None else prompt_configs
        )
        # Thematische Kategorie je Artikel-ID (Spalte Article_Type im CSV-Schema)
        self.article_types = dict(article_types) if article_types else {}
        self.seeds = list(seeds) if seeds is not None else list(self.DEFAULT_SEEDS)
        # Preise je 1 Mio. Tokens: {model_name: {"input": float, "output": float}}
        self.prices = dict(prices) if prices else {}
        # Zeilen der aktuellen Sitzung (Quelle der Wahrheit sind die CSV-Dateien)
        self.results = []

    def set_prompt_config(self, prompt_config):
        """Ersetzt die globale Prompt-Konfiguration."""
        self.prompt_config = self._normalize_prompt_config(prompt_config)

    def set_prompt_configs(self, prompt_configs):
        """Ersetzt die Prompt-Konfigurationen je Artikel-ID."""
        self.article_prompt_configs = self._normalize_prompt_config_mapping(prompt_configs)

    def set_article_prompt_configs(self, article_prompt_configs):
        """Ersetzt die Prompt-Konfigurationen je Artikel-ID."""
        self.article_prompt_configs = self._normalize_prompt_config_mapping(article_prompt_configs)

    # ------------------------------------------------------------------
    # Versuchssteuerung
    # ------------------------------------------------------------------

    def run_replication(self, run_index, dry_run=False, force=False, output_dir="../results"):
        """
        Führt genau einen Durchlauf aus (Durchlauf N nutzt Seed[N-1]).

        Idempotent: Ist der Durchlauf vollständig, passiert nichts; ist er
        unvollständig (Abbruch, fehlgeschlagene Aufrufe), werden ausschließlich
        die fehlenden Variante-Modell-Kombinationen nachgeholt. Ein bewusstes
        Neuerheben erfordert force=True. Dry Runs (kostenlos) starten immer
        frisch und schreiben in separate Dateien mit dem Suffix _DRYRUN.
        """
        if not 1 <= run_index <= len(self.seeds):
            raise ValueError(f"run_index muss zwischen 1 und {len(self.seeds)} liegen.")
        seed = self.seeds[run_index - 1]
        os.makedirs(output_dir, exist_ok=True)
        results_file, errors_file = self._run_files(run_index, seed, output_dir, dry_run)

        if os.path.exists(results_file) and (dry_run or force):
            os.remove(results_file)

        conditions = self._collect_conditions()
        expected_calls = len(conditions) * len(self.connectors)
        done = self._read_done_keys(results_file)

        todo = [
            (condition, connector, self._model_name(connector))
            for condition in conditions
            for connector in self.connectors
            if (condition["article_id"], condition["method"], self._model_name(connector)) not in done
        ]

        if not todo:
            print(
                f"Durchlauf {run_index} (Seed {seed}) ist bereits vollständig "
                f"({expected_calls} Antworten in {os.path.basename(results_file)}) -- übersprungen. "
                f"Neu erheben nur mit force=True."
            )
            return

        mode = "DRY RUN, keine API-Aufrufe" if dry_run else "ECHTLAUF, API-Aufrufe verursachen Kosten"
        resumed = f"; {len(done)} vorhandene Antworten werden übernommen" if done else ""
        print(
            f"--- Durchlauf {run_index} (Seed {seed}): {len(todo)} von {expected_calls} "
            f"Aufrufen ausstehend ({mode}{resumed}) ---"
        )

        new_rows = []
        error_count = 0
        for call_no, (condition, connector, model_name) in enumerate(todo, start=1):
            print(f"[{call_no}/{len(todo)}] {condition['filename']} | {model_name}")

            row = {
                "Timestamp": datetime.now().isoformat(),
                "Article_ID": condition["article_id"],
                "Article_Type": self.article_types.get(condition["article_id"], ""),
                "Method": condition["method"],
                "Seed": seed,
                "Model": model_name,
                "Model_Version": "",
                "System_Prompt": condition["system_prompt"],
                "User_Prompt": condition["user_prompt"],
                "Answer": "",
                "Prompt_Tokens": "",
                "Completion_Tokens": "",
                "Total_Tokens": "",
                "Run": run_index,
            }

            if dry_run:
                row["Answer"] = "[DRY RUN] Es wurde keine API-Anfrage gesendet."
                self._append_csv_row(results_file, self.CSV_COLUMNS, row)
                new_rows.append(row)
            else:
                try:
                    result = connector.send_query(
                        condition["system_prompt"], condition["user_prompt"], seed=seed
                    )
                    row["Answer"] = result["text"]
                    for result_key, column in (
                        ("model_version", "Model_Version"),
                        ("prompt_tokens", "Prompt_Tokens"),
                        ("completion_tokens", "Completion_Tokens"),
                        ("total_tokens", "Total_Tokens"),
                    ):
                        if result.get(result_key) is not None:
                            row[column] = result[result_key]
                    self._append_csv_row(results_file, self.CSV_COLUMNS, row)
                    new_rows.append(row)
                except Exception as exc:
                    # Fehlgeschlagene Aufrufe gehören nicht in den Ergebnisdatensatz;
                    # sie werden separat protokolliert und beim nächsten Aufruf nachgeholt.
                    error_count += 1
                    row["Error"] = str(exc)
                    self._append_csv_row(errors_file, self.CSV_COLUMNS + ["Error"], row)
                    print(f"[FEHLER] {condition['filename']} | {model_name}: {exc}")

            self.results.append(row)

        rows_now = len(self._read_rows(results_file))
        print(
            f"--- Durchlauf {run_index}: {len(new_rows)} neue Antworten, {error_count} Fehler; "
            f"Datei enthält {rows_now}/{expected_calls} Antworten ---"
        )
        if not dry_run and new_rows:
            prompt_sum = sum(self._to_int(r.get("Prompt_Tokens")) or 0 for r in new_rows)
            completion_sum = sum(self._to_int(r.get("Completion_Tokens")) or 0 for r in new_rows)
            total_sum = sum(self._to_int(r.get("Total_Tokens")) or 0 for r in new_rows)
            summary = (
                f"Token-Verbrauch dieser Sitzung: {prompt_sum:,} Input + "
                f"{completion_sum:,} Output = {total_sum:,} gesamt"
            )
            session_costs = [c for c in (self._row_cost(r) for r in new_rows) if c is not None]
            if session_costs:
                summary += f" (~{sum(session_costs):.4f} laut Preisliste)"
            print(summary)
        if not dry_run and rows_now < expected_calls:
            print(
                f"[HINWEIS] {expected_calls - rows_now} Aufrufe fehlen weiterhin "
                f"(siehe {os.path.basename(errors_file)}); erneutes "
                f"run_replication({run_index}) holt genau diese nach."
            )

    def run_full_experiment(self, seeds=None, dry_run=True, output_dir="../results", prompt_config=None):
        """
        Führt alle Durchläufe nacheinander aus. Dank der Idempotenz von
        run_replication werden nur fehlende Antworten erhoben; ein erneuter
        Aufruf vervollständigt also lediglich das Experiment.
        """
        if prompt_config is not None:
            self.set_prompt_config(prompt_config)
        if seeds is not None:
            self.seeds = list(seeds)

        self.results = []
        expected_calls = len(self._collect_conditions()) * len(self.connectors)
        mode = "DRY RUN, keine API-Aufrufe" if dry_run else "ECHTLAUF, API-Aufrufe verursachen Kosten"
        print(
            f"--- Experiment gestartet am {datetime.now().strftime('%d.%m.%Y %H:%M')} ({mode}): "
            f"{len(self.seeds)} Durchläufe x {expected_calls} Aufrufe ---"
        )
        for run_index in range(1, len(self.seeds) + 1):
            self.run_replication(run_index, dry_run=dry_run, output_dir=output_dir)
        print(f"--- Experiment beendet: {len(self.results)} Aufrufe in dieser Sitzung ---")

    # ------------------------------------------------------------------
    # Kosten- und Fortschrittskontrolle
    # ------------------------------------------------------------------

    def status(self, output_dir="../results"):
        """
        Fortschritts- und Kostenübersicht über alle Durchläufe auf Basis der
        CSV-Dateien (Echtläufe, ohne _DRYRUN). Gibt einen DataFrame zurück.
        """
        expected_calls = len(self._collect_conditions()) * len(self.connectors)
        overview = []
        any_cost = False
        for run_index, seed in enumerate(self.seeds, start=1):
            results_file, errors_file = self._run_files(run_index, seed, output_dir, dry_run=False)
            rows = self._read_rows(results_file)
            row_costs = [c for c in (self._row_cost(r) for r in rows) if c is not None]
            run_cost = sum(row_costs) if row_costs else None
            any_cost = any_cost or run_cost is not None
            overview.append({
                "Durchlauf": run_index,
                "Seed": seed,
                "Antworten": len(rows),
                "Fehlend": expected_calls - len(rows),
                "Fehler-Log": len(self._read_rows(errors_file)),
                "Prompt_Tokens": sum(self._to_int(r.get("Prompt_Tokens")) or 0 for r in rows),
                "Completion_Tokens": sum(self._to_int(r.get("Completion_Tokens")) or 0 for r in rows),
                "Total_Tokens": sum(self._to_int(r.get("Total_Tokens")) or 0 for r in rows),
                "Kosten": round(run_cost, 4) if run_cost is not None else None,
            })

        df = pd.DataFrame(overview)
        if not any_cost:
            df = df.drop(columns=["Kosten"])
        total_answers = int(df["Antworten"].sum())
        summary = (
            f"Gesamt: {total_answers}/{expected_calls * len(self.seeds)} Antworten, "
            f"{int(df['Total_Tokens'].sum()):,} Tokens"
        )
        if any_cost:
            summary += f", Kosten bisher: {df['Kosten'].sum():.4f} (laut Preisliste)"
        else:
            summary += " (keine Preisliste gesetzt -- nur Token-Zählung)"
        print(summary)
        return df

    def estimate(self, output_dir="../results"):
        """
        Kostenschätzung ohne API-Aufrufe: Input-Tokens werden über die
        Zeichen/4-Heuristik geschätzt; sobald mindestens ein Durchlauf
        vollständig vorliegt, werden die restlichen Durchläufe aus dessen
        Ist-Werten hochgerechnet (inklusive Reasoning-Tokens im Output).
        """
        conditions = self._collect_conditions()
        expected_calls = len(conditions) * len(self.connectors)
        prompt_chars = sum(
            len(c["system_prompt"]) + len(c["user_prompt"]) for c in conditions
        ) * len(self.connectors)
        estimated_input = prompt_chars // 4

        completed_rows = []
        runs_completed = 0
        for run_index, seed in enumerate(self.seeds, start=1):
            results_file, _ = self._run_files(run_index, seed, output_dir, dry_run=False)
            rows = self._read_rows(results_file)
            if len(rows) == expected_calls:
                runs_completed += 1
                completed_rows.extend(rows)
        runs_remaining = len(self.seeds) - runs_completed

        print(f"Aufrufe je Durchlauf: {expected_calls}  |  Durchläufe: {len(self.seeds)} "
              f"({runs_completed} vollständig, {runs_remaining} ausstehend)")
        print(f"Geschätzte Input-Tokens je Durchlauf (Zeichen/4-Heuristik): ~{estimated_input:,}")

        estimate_result = {
            "calls_per_run": expected_calls,
            "estimated_input_tokens_per_run": estimated_input,
            "runs_completed": runs_completed,
            "runs_remaining": runs_remaining,
        }

        if completed_rows:
            prompt_values = [self._to_int(r.get("Prompt_Tokens")) for r in completed_rows]
            completion_values = [self._to_int(r.get("Completion_Tokens")) for r in completed_rows]
            prompt_values = [v for v in prompt_values if v is not None]
            completion_values = [v for v in completion_values if v is not None]
            if prompt_values and completion_values:
                avg_prompt = sum(prompt_values) / len(prompt_values)
                avg_completion = sum(completion_values) / len(completion_values)
                projected_run = (avg_prompt + avg_completion) * expected_calls
                print(
                    f"Ist-Werte aus {runs_completed} vollständigen Durchläufen: "
                    f"~{avg_prompt:,.0f} Input- + ~{avg_completion:,.0f} Output-Tokens je Aufruf"
                )
                print(
                    f"Hochrechnung je weiterem Durchlauf: ~{projected_run:,.0f} Tokens; "
                    f"für {runs_remaining} ausstehende: ~{projected_run * runs_remaining:,.0f} Tokens"
                )
                row_costs = [c for c in (self._row_cost(r) for r in completed_rows) if c is not None]
                if row_costs:
                    cost_per_run = sum(row_costs) / runs_completed
                    print(
                        f"Kosten laut Preisliste: ~{cost_per_run:.4f} je Durchlauf, "
                        f"~{cost_per_run * runs_remaining:.4f} für die ausstehenden"
                    )
                    estimate_result["projected_cost_per_run"] = cost_per_run
                estimate_result["actual_avg_prompt_tokens"] = avg_prompt
                estimate_result["actual_avg_completion_tokens"] = avg_completion
                estimate_result["projected_tokens_per_run"] = projected_run
        else:
            print("Output-Tokens: vor dem ersten vollständigen Echtlauf nicht prognostizierbar "
                  "(Reasoning-Tokens sind nur im Ist-Verbrauch sichtbar).")

        return estimate_result

    def save_results(self, output_file="../results/experiment_results_all.csv"):
        """Speichert alle Zeilen der aktuellen Sitzung zusätzlich als Gesamt-CSV."""
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False, sep=";", encoding="utf-8")
        print(f"Ergebnisse erfolgreich in {output_file} gespeichert!")

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------

    def _collect_conditions(self):
        """Lädt alle Artikelvarianten und rendert die Prompts (offline, ohne API)."""
        conditions = []
        for filename in sorted(self.loader.list_available_articles(self.article_folder)):
            content = self._load_article(filename)
            article_id, method = self._parse_filename(filename)
            config = self._get_prompt_config(article_id)
            self._ensure_article_placeholder(config)
            conditions.append({
                "filename": filename,
                "article_id": article_id,
                "method": method,
                "system_prompt": self._render_prompt(
                    config["system_prompt"], content, article_id, method, filename
                ),
                "user_prompt": self._render_prompt(
                    config["user_prompt"], content, article_id, method, filename
                ),
            })
        return conditions

    @staticmethod
    def _run_files(run_index, seed, output_dir, dry_run):
        """Stabile, idempotente Dateinamen je Durchlauf (kein Zeitstempel)."""
        suffix = "_DRYRUN" if dry_run else ""
        base = f"run{run_index}_seed{seed}{suffix}.csv"
        return (
            os.path.join(output_dir, f"experiment_results_{base}"),
            os.path.join(output_dir, f"experiment_errors_{base}"),
        )

    @staticmethod
    def _read_rows(file_path):
        """Liest eine Ergebnis-/Fehlerdatei als Liste von Dicts (leer, falls fehlend)."""
        if not os.path.exists(file_path):
            return []
        with open(file_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=";"))

    def _read_done_keys(self, results_file):
        """Bereits erhobene (Artikel, Methode, Modell)-Kombinationen eines Durchlaufs."""
        return {
            (row.get("Article_ID"), row.get("Method"), row.get("Model"))
            for row in self._read_rows(results_file)
        }

    @staticmethod
    def _model_name(connector):
        return getattr(
            connector,
            "model_name",
            getattr(connector, "provider", connector.__class__.__name__),
        )

    @staticmethod
    def _to_int(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _row_cost(self, row):
        """Kosten einer Zeile laut Preisliste (je 1 Mio. Tokens); None, wenn nicht berechenbar."""
        price = self.prices.get(row.get("Model"))
        prompt_tokens = self._to_int(row.get("Prompt_Tokens"))
        completion_tokens = self._to_int(row.get("Completion_Tokens"))
        if (
            not price
            or price.get("input") is None
            or price.get("output") is None
            or prompt_tokens is None
            or completion_tokens is None
        ):
            return None
        return prompt_tokens / 1e6 * price["input"] + completion_tokens / 1e6 * price["output"]

    @staticmethod
    def _append_csv_row(file_path, columns, row):
        """Hängt eine Ergebniszeile an die CSV an; beim ersten Aufruf mit Header."""
        write_header = not os.path.exists(file_path)
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";", extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _ensure_article_placeholder(prompt_config):
        """Stellt sicher, dass der Artikeltext das Modell überhaupt erreicht."""
        combined = prompt_config["system_prompt"] + prompt_config["user_prompt"]
        if "{article}" not in combined and "{content}" not in combined:
            raise ValueError(
                "Weder System- noch User-Prompt enthält den Platzhalter {article} – "
                "der Artikeltext würde das Modell nie erreichen."
            )

    def _load_article(self, filename):
        """Lädt eine HTML-Datei aus dem konfigurierten Artikelordner."""
        if self.article_folder == "processed":
            return self.loader.load_processed(filename)

        folder_paths = {
            "raw": self.loader.raw_path,
        }
        folder_path = folder_paths.get(self.article_folder, self.article_folder)
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artikeldatei nicht gefunden: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _parse_filename(filename):
        """Extrahiert Artikel-ID und Methodenvariante aus dem Dateinamen (z. B. 01_Original)."""
        name = os.path.splitext(filename)[0]
        parts = name.split("_", 1)
        article_id = parts[0]
        method = parts[1] if len(parts) > 1 else "Unbekannt"
        return article_id, method

    def _get_prompt_config(self, article_id):
        """
        Wählt die passende Prompt-Konfiguration.

        Priorität:
            1. Artikel-ID
            2. globale Prompt-Konfiguration
        """
        return self.article_prompt_configs.get(article_id, self.prompt_config)

    @staticmethod
    def _render_prompt(template, content, article_id, method, filename):
        """Setzt Artikeldaten in ein Prompt-Template ein."""
        replacements = {
            "{article}": content,
            "{content}": content,
            "{article_id}": article_id,
            "{article_type}": method,
            "{filename}": filename,
        }

        prompt = template
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)

        return prompt

    @classmethod
    def _normalize_prompt_config(cls, prompt_config):
        """Validiert und normalisiert ein System/User-Prompt-Paar."""
        if prompt_config is None:
            return {
                "system_prompt": cls.DEFAULT_SYSTEM_PROMPT,
                "user_prompt": cls.DEFAULT_USER_PROMPT,
            }

        if isinstance(prompt_config, str):
            prompt_config = {
                "system_prompt": cls.DEFAULT_SYSTEM_PROMPT,
                "user_prompt": prompt_config,
            }

        if isinstance(prompt_config, (list, tuple)) and len(prompt_config) == 2:
            prompt_config = {
                "system_prompt": prompt_config[0],
                "user_prompt": prompt_config[1],
            }

        if not isinstance(prompt_config, dict):
            raise TypeError(
                "Prompt-Konfigurationen müssen dict, str oder ein Tupel "
                "(system_prompt, user_prompt) sein."
            )

        system_prompt = prompt_config.get("system_prompt", prompt_config.get("system"))
        user_prompt = prompt_config.get(
            "user_prompt",
            prompt_config.get("prompt", prompt_config.get("normal_prompt")),
        )

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("Jede Prompt-Konfiguration braucht einen system_prompt.")

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("Jede Prompt-Konfiguration braucht einen user_prompt.")

        return {
            "system_prompt": system_prompt.strip(),
            "user_prompt": user_prompt.strip(),
        }

    @classmethod
    def _normalize_prompt_config_mapping(cls, prompt_config_mapping):
        """Validiert und normalisiert Prompt-Konfigurationen nach Artikel-ID."""
        if prompt_config_mapping is None:
            return {}

        if not isinstance(prompt_config_mapping, dict):
            raise TypeError("Prompt-Konfigurations-Mappings müssen Dictionaries sein.")

        normalized = {}
        for key, prompt_config in prompt_config_mapping.items():
            key = str(key).strip()
            if not key:
                raise ValueError("Prompt-Konfigurations-Keys dürfen nicht leer sein.")

            normalized[key] = cls._normalize_prompt_config(prompt_config)

        return normalized
