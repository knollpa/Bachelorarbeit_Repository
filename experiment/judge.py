import csv
import json
import os
import subprocess
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------------------
# Bewertungsschema und Prompt (Abschnitt "Evaluierungsverfahren" der Thesis)
# ----------------------------------------------------------------------

# JSON-Schema für die strukturierte Ausgabe des Evaluators. Es spiegelt das
# Bewertungsmodell aus Abschnitt "Metriken und Scorecard": einmalige Zerlegung
# in atomare Aussagen, disjunkte Zuordnung zu genau einer Kategorie sowie
# binäre Kernaussagen-Urteile. Die Metrikwerte selbst liefert der Evaluator
# bewusst nicht -- sie werden deterministisch in Python berechnet.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["correct", "intrinsic_error", "extrinsic"],
                    },
                },
                "required": ["claim", "category"],
                "additionalProperties": False,
            },
        },
        "key_messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "covered": {"type": "boolean"},
                },
                "required": ["id", "covered"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims", "key_messages"],
    "additionalProperties": False,
}

# Kombinierter Bewertungsprompt: erhebt alle drei Metriken in einem Aufruf.
# Englischsprachig, konsistent mit den Prompts der Versuchsdurchführung; die
# Kategoriendefinitionen entsprechen inhaltlich dem Klassifikationsschema aus
# Abschnitt "Metriken und Scorecard" der Thesis und dem deutschsprachigen
# Codebuch der menschlichen Rater (correct = korrekt, intrinsic_error =
# intrinsisch fehlerhaft, extrinsic = extrinsisch). Der Wortlaut steht in
# Abschnitt "Evaluierungsverfahren" der Thesis.
JUDGE_SYSTEM_PROMPT = (
    "You are a precise, objective evaluator of the quality of LLM answers to press "
    "articles. You apply only the provided definitions and judge solely on the basis "
    "of the provided source text. Do not use your own world knowledge to decide "
    "whether a statement is true or false."
)

JUDGE_USER_PROMPT_TEMPLATE = """Evaluate the following LLM answer against the source text and the key messages.

## Step 1: Decomposition into atomic claims
Decompose the LLM answer completely into atomic claims. An atomic claim is the smallest independently verifiable factual statement (e.g., a number, a date, a proper name, a model designation, or a single factual assertion). Pure meta statements without factual content (e.g., "The article describes ...") are not atomic claims.

## Step 2: Disjoint classification
Assign each atomic claim to exactly one of the three categories:
- "correct": The claim is supported by the provided source text.
- "intrinsic_error": The claim contradicts a verifiable fact of the source text.
- "extrinsic": The claim is neither supported nor contradicted by the source text and presumably stems from the model's parametric knowledge.

## Step 3: Key message coverage
For each numbered key message, check whether the LLM answer conveys it without material omission ("covered": true) or not ("covered": false). Assess only the presence, not the factual correctness of the rendition.

## Output format
Respond exclusively with a JSON object (no Markdown, no explanations):
{{"claims": [{{"claim": "<atomic claim>", "category": "correct|intrinsic_error|extrinsic"}}, ...], "key_messages": [{{"id": 1, "covered": true}}, ...]}}
The "key_messages" field must contain exactly one judgment for each numbered key message.

## Source text (press article)
[ARTICLE START]
{article}
[ARTICLE END]

## Key messages
{key_messages}

## LLM answer to evaluate
[ANSWER START]
{answer}
[ANSWER END]"""


# ----------------------------------------------------------------------
# Auswertung des Rohurteils
# ----------------------------------------------------------------------

VALID_CATEGORIES = {"correct", "intrinsic_error", "extrinsic"}


def parse_judgment(text, expected_km_count):
    """
    Parst und validiert das Rohurteil des Evaluators.

    Toleriert Markdown-Codefences und umgebenden Text (relevant für die
    Zweitmeinungs-Evaluatoren ohne anbieterseitig erzwungenes Schema).
    Wirft ValueError bei jeder Abweichung vom Schema, damit fehlerhafte
    Urteile nie in den Ergebnisdatensatz gelangen.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Kein JSON-Objekt in der Evaluator-Antwort gefunden.")
    judgment = json.loads(cleaned[start : end + 1])

    claims = judgment.get("claims")
    key_messages = judgment.get("key_messages")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Urteil enthält keine Liste atomarer Aussagen (claims).")
    for claim in claims:
        if claim.get("category") not in VALID_CATEGORIES:
            raise ValueError(f"Ungültige Kategorie im Urteil: {claim.get('category')!r}")
        if not str(claim.get("claim", "")).strip():
            raise ValueError("Leere atomare Aussage im Urteil.")
    if not isinstance(key_messages, list):
        raise ValueError("Urteil enthält keine Kernaussagen-Liste (key_messages).")
    judged_ids = sorted(int(km["id"]) for km in key_messages)
    if judged_ids != list(range(1, expected_km_count + 1)):
        raise ValueError(
            f"Kernaussagen-Urteile unvollständig: erwartet 1..{expected_km_count}, "
            f"erhalten {judged_ids}."
        )
    for km in key_messages:
        if not isinstance(km.get("covered"), bool):
            raise ValueError("Kernaussagen-Urteil 'covered' ist kein Boolean.")
    return judgment


def compute_metrics(judgment):
    """
    Berechnet IA, HR und KMC deterministisch aus den Klassifikationen,
    exakt nach den Gleichungen aus Abschnitt "Metriken und Scorecard":
        IA  = korrekt / (korrekt + intrinsisch) * 100   (None bei Nenner 0)
        HR  = extrinsisch / gesamt * 100
        KMC = abgedeckte / alle Kernaussagen * 100
    """
    counts = {category: 0 for category in VALID_CATEGORIES}
    for claim in judgment["claims"]:
        counts[claim["category"]] += 1
    korrekt = counts["correct"]
    intrinsisch = counts["intrinsic_error"]
    extrinsisch = counts["extrinsic"]
    total = korrekt + intrinsisch + extrinsisch

    km_total = len(judgment["key_messages"])
    km_covered = sum(1 for km in judgment["key_messages"] if km["covered"])

    ia_denominator = korrekt + intrinsisch
    return {
        "Claims_Korrekt": korrekt,
        "Claims_Intrinsisch": intrinsisch,
        "Claims_Extrinsisch": extrinsisch,
        "Claims_Gesamt": total,
        "KM_Covered": km_covered,
        "KM_Gesamt": km_total,
        "IA": round(korrekt / ia_denominator * 100, 2) if ia_denominator else None,
        "HR": round(extrinsisch / total * 100, 2) if total else None,
        "KMC": round(km_covered / km_total * 100, 2) if km_total else None,
    }


# ----------------------------------------------------------------------
# Anbindung des Evaluators (Anthropic-API)
# ----------------------------------------------------------------------

class JudgeConnector:
    """
    Kapselt die Kommunikation mit dem Evaluatormodell Claude Opus 4.8.

    Die Anthropic-API stellt für Opus 4.8 keine Sampling-Parameter und keinen
    Seed bereit (Abschnitt "Evaluierungsverfahren" der Thesis); die
    Antwortstruktur wird über das JSON-Schema anbieterseitig erzwungen
    (Structured Output). Die tatsächlich servierte Modellversion wird je
    Bewertung zurückgegeben und protokolliert.
    """

    MODEL_ID = "claude-opus-4-8"

    def __init__(self, model_name=None, max_tokens=16000):
        import anthropic  # lazy: Tests mit Fake-Connectors brauchen das Paket nicht

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY fehlt!")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name or self.MODEL_ID
        self.max_tokens = max_tokens

    def send_query(self, system_prompt, user_prompt, max_retries=3, backoff_seconds=2.0):
        """Bewertungsaufruf mit Retries; Rückgabeformat wie LLMConnector.send_query."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._call(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait_seconds = backoff_seconds * 2 ** (attempt - 1)
                    print(
                        f"[WARNUNG] {self.model_name}: Versuch {attempt}/{max_retries} "
                        f"fehlgeschlagen ({exc}). Nächster Versuch in {wait_seconds:.0f}s."
                    )
                    time.sleep(wait_seconds)
        raise RuntimeError(
            f"{self.model_name}: Bewertung nach {max_retries} Versuchen fehlgeschlagen: {last_error}"
        ) from last_error

    def _call(self, system_prompt, user_prompt):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Evaluator hat die Bewertung verweigert (stop_reason=refusal).")
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            raise RuntimeError("Leere Antwort vom Evaluator erhalten.")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_version": getattr(response, "model", None),
        }


class ClaudeCLIJudgeConnector:
    """
    Ruft den Evaluator über die Claude-Code-CLI im nicht-interaktiven
    Headless-Modus auf (claude -p). Die Bewertung wird damit über das
    Claude-Abo des CLI-Logins abgerechnet statt über API-Guthaben.

    Konfiguration des Aufrufs (Abschnitt "Evaluierungsverfahren" der Thesis):
        - --system-prompt ersetzt den Standard-Systemprompt der CLI
          vollständig durch den Judge-System-Prompt,
        - --tools "" deaktiviert sämtliche Werkzeuge (genau ein Modellaufruf
          je Bewertung, keine Agenten-Schleife),
        - --json-schema erzwingt die Antwortstruktur,
        - --output-format json liefert Ergebnis, Token-Verbrauch und die
          servierte Modellversion als auswertbaren JSON-Envelope.

    ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN werden aus der Subprozess-Umgebung
    entfernt, damit die CLI nie versehentlich per API-Key abrechnet.
    """

    MODEL_ID = "claude-opus-4-8"

    def __init__(self, model_name=None, cli_command="claude",
                 timeout_seconds=600, pause_seconds=2.0):
        self.model_name = model_name or self.MODEL_ID
        self.cli_command = cli_command
        self.timeout_seconds = timeout_seconds
        # Kurze Pause nach jedem erfolgreichen Aufruf als Rücksicht auf die
        # Abo-Rate-Limits; Drosselungen sind dank Resume ohnehin unkritisch.
        self.pause_seconds = pause_seconds

    def send_query(self, system_prompt, user_prompt, max_retries=3, backoff_seconds=2.0):
        """Bewertungsaufruf mit Retries; Rückgabeformat wie JudgeConnector.send_query."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self._call(system_prompt, user_prompt)
                if self.pause_seconds:
                    time.sleep(self.pause_seconds)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait_seconds = backoff_seconds * 2 ** (attempt - 1)
                    print(
                        f"[WARNUNG] {self.cli_command} -p: Versuch {attempt}/{max_retries} "
                        f"fehlgeschlagen ({exc}). Nächster Versuch in {wait_seconds:.0f}s."
                    )
                    time.sleep(wait_seconds)
        raise RuntimeError(
            f"{self.cli_command} -p: Bewertung nach {max_retries} Versuchen fehlgeschlagen: {last_error}"
        ) from last_error

    def _call(self, system_prompt, user_prompt):
        completed = subprocess.run(
            self.build_command(system_prompt),
            input=user_prompt,
            env=self.build_env(),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"claude -p beendet mit Exit {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[:500]}"
            )
        return self.parse_cli_output(completed.stdout)

    def build_command(self, system_prompt):
        """Baut den CLI-Aufruf; der User-Prompt wird über stdin übergeben."""
        return [
            self.cli_command,
            "-p",
            "--model", self.model_name,
            "--system-prompt", system_prompt,
            "--json-schema", json.dumps(JUDGE_SCHEMA),
            "--output-format", "json",
            "--tools", "",
        ]

    @staticmethod
    def build_env():
        """Subprozess-Umgebung ohne API-Zugangsdaten (Abo-Login der CLI greift)."""
        env = dict(os.environ)
        for variable in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(variable, None)
        return env

    def parse_cli_output(self, stdout):
        """Wertet den JSON-Envelope von --output-format json aus."""
        envelope = json.loads(stdout)
        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude -p meldet Fehler: {envelope.get('subtype') or envelope.get('result')}"
            )

        text = envelope.get("structured_output")
        if text is None:
            text = envelope.get("result")
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False)
        if not text:
            raise RuntimeError("Leere Antwort von claude -p erhalten.")

        usage = envelope.get("usage") or {}
        prompt_tokens = usage.get("input_tokens")
        if prompt_tokens is not None:
            # Cache-Anteile zählen zum tatsächlichen Prompt-Umfang dazu.
            prompt_tokens += (usage.get("cache_creation_input_tokens") or 0)
            prompt_tokens += (usage.get("cache_read_input_tokens") or 0)
        completion_tokens = usage.get("output_tokens")
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        model_version = None
        model_usage = envelope.get("modelUsage")
        if isinstance(model_usage, dict) and model_usage:
            model_version = sorted(model_usage)[0]

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_version": model_version or self.model_name,
        }


# ----------------------------------------------------------------------
# Steuerung der Bewertung
# ----------------------------------------------------------------------

class JudgeRunner:
    """
    Bewertet die Experiment-Antworten mit dem LLM-as-a-Judge-Verfahren.

    Kompakt-Design gemäß Thesis: ein Bewertungsaufruf je Antwort erhebt alle
    drei Metriken (gemeinsame Analyseeinheit), ein einziger Durchgang je
    Antwort. Idempotent wie der ExperimentRunner: vorhandene Bewertungen
    werden übersprungen, Fehler landen in einer separaten Fehlerdatei und
    werden beim nächsten Aufruf nachgeholt.
    """

    CSV_COLUMNS = [
        "Timestamp",
        "Article_ID",
        "Article_Type",
        "Method",
        "Seed",
        "Model",
        "Judge_Model",
        "Judge_Model_Version",
        "Claims_Korrekt",
        "Claims_Intrinsisch",
        "Claims_Extrinsisch",
        "Claims_Gesamt",
        "KM_Covered",
        "KM_Gesamt",
        "IA",
        "HR",
        "KMC",
        "Judge_JSON",
        "Prompt_Tokens",
        "Completion_Tokens",
        "Total_Tokens",
    ]

    # Preise je 1 Mio. Tokens (Claude Opus 4.8, Stand Juli 2026) für estimate/status.
    DEFAULT_PRICES = {"input": 5.0, "output": 25.0}

    def __init__(
        self,
        loader,
        connector,
        results_dir="../results",
        key_messages_path="../data/annotations/key_messages.json",
        prices=None,
        system_prompt=None,
        user_prompt_template=None,
    ):
        self.loader = loader
        self.connector = connector
        self.results_dir = results_dir
        self.prices = dict(prices) if prices else dict(self.DEFAULT_PRICES)
        self.system_prompt = system_prompt or JUDGE_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or JUDGE_USER_PROMPT_TEMPLATE
        with open(key_messages_path, encoding="utf-8") as f:
            self.key_messages = json.load(f)

    # ------------------------------------------------------------------
    # Öffentliche Steuerung
    # ------------------------------------------------------------------

    def run(self, keys=None, dry_run=False, force=False, output_name="judge_results"):
        """
        Bewertet alle (oder die per keys eingeschränkten) Experiment-Antworten.

        keys: optionale Menge von (Article_ID, Method, Model, Seed)-Tupeln
              (Seed als String), z. B. die Validierungsstichprobe für den Pilot.
        """
        answers = self.load_answers()
        if keys is not None:
            keys = {self._normalize_key(k) for k in keys}
            answers = [row for row in answers if self._condition_key(row) in keys]
            missing = keys - {self._condition_key(row) for row in answers}
            if missing:
                raise ValueError(f"Keine Experiment-Antworten zu diesen Schlüsseln: {sorted(missing)}")

        results_file, errors_file = self._output_files(output_name, dry_run)
        if os.path.exists(results_file) and (dry_run or force):
            os.remove(results_file)

        done = self._read_done_keys(results_file)
        todo = [row for row in answers if self._condition_key(row) not in done]

        if not todo:
            print(
                f"Bewertung ist bereits vollständig ({len(answers)} Urteile in "
                f"{os.path.basename(results_file)}) -- übersprungen. Neu bewerten nur mit force=True."
            )
            return

        mode = "DRY RUN, keine API-Aufrufe" if dry_run else "ECHTLAUF, API-Aufrufe verursachen Kosten"
        resumed = f"; {len(done)} vorhandene Urteile werden übernommen" if done else ""
        print(f"--- Judge: {len(todo)} von {len(answers)} Bewertungen ausstehend ({mode}{resumed}) ---")

        error_count = 0
        new_rows = []
        for call_no, answer_row in enumerate(todo, start=1):
            key = self._condition_key(answer_row)
            print(f"[{call_no}/{len(todo)}] {key[0]}_{key[1]} | {key[2]} | Seed {key[3]}")
            row = self._base_row(answer_row)

            if dry_run:
                row["Judge_JSON"] = "[DRY RUN] Es wurde keine API-Anfrage gesendet."
                self._append_csv_row(results_file, self.CSV_COLUMNS, row)
                new_rows.append(row)
                continue

            try:
                user_prompt = self.build_user_prompt(answer_row)
                result = self.connector.send_query(self.system_prompt, user_prompt)
                judgment = parse_judgment(
                    result["text"], self._km_count(answer_row["Article_ID"])
                )
                row.update(compute_metrics(judgment))
                row["Judge_JSON"] = json.dumps(judgment, ensure_ascii=False)
                for result_key, column in (
                    ("model_version", "Judge_Model_Version"),
                    ("prompt_tokens", "Prompt_Tokens"),
                    ("completion_tokens", "Completion_Tokens"),
                    ("total_tokens", "Total_Tokens"),
                ):
                    if result.get(result_key) is not None:
                        row[column] = result[result_key]
                self._append_csv_row(results_file, self.CSV_COLUMNS, row)
                new_rows.append(row)
            except Exception as exc:
                error_count += 1
                row["Error"] = str(exc)
                self._append_csv_row(errors_file, self.CSV_COLUMNS + ["Error"], row)
                print(f"[FEHLER] {key}: {exc}")

        rows_now = len(self._read_rows(results_file))
        print(
            f"--- Judge: {len(new_rows)} neue Urteile, {error_count} Fehler; "
            f"Datei enthält {rows_now}/{len(answers)} Urteile ---"
        )
        if not dry_run and new_rows:
            self._print_session_costs(new_rows)
        if not dry_run and error_count:
            print(
                f"[HINWEIS] Fehlgeschlagene Bewertungen siehe {os.path.basename(errors_file)}; "
                f"erneutes run() holt genau diese nach."
            )

    def run_sample(self, sample_file="validation_sample.csv", dry_run=False, force=False,
                   output_name="judge_pilot", connector=None, judge_label=None):
        """
        Bewertet ausschließlich die Validierungsstichprobe (Pilot vor dem
        Volllauf). Über connector/judge_label kann derselbe Ablauf für die
        Zweitmeinungs-Evaluatoren wiederverwendet werden.
        """
        sample_path = os.path.join(self.results_dir, sample_file)
        keys = {
            (row["Article_ID"], row["Method"], row["Model"], row["Seed"])
            for row in self._read_rows(sample_path)
        }
        if not keys:
            raise FileNotFoundError(
                f"Validierungsstichprobe nicht gefunden oder leer: {sample_path}"
            )
        original_connector = self.connector
        try:
            if connector is not None:
                self.connector = connector
            suffix = f"_{judge_label}" if judge_label else ""
            self.run(keys=keys, dry_run=dry_run, force=force, output_name=f"{output_name}{suffix}")
        finally:
            self.connector = original_connector

    def status(self, output_name="judge_results"):
        """Fortschritts- und Kostenübersicht der Bewertung als DataFrame."""
        answers = self.load_answers()
        results_file, errors_file = self._output_files(output_name, dry_run=False)
        rows = self._read_rows(results_file)
        judged_keys = {self._condition_key(r) for r in rows}
        versions = sorted({r.get("Judge_Model_Version", "") for r in rows if r.get("Judge_Model_Version")})

        overview = pd.DataFrame([{
            "Antworten_gesamt": len(answers),
            "Bewertet": len(rows),
            "Fehlend": len(answers) - len(judged_keys),
            "Fehler-Log": len(self._read_rows(errors_file)),
            "Prompt_Tokens": sum(self._to_int(r.get("Prompt_Tokens")) or 0 for r in rows),
            "Completion_Tokens": sum(self._to_int(r.get("Completion_Tokens")) or 0 for r in rows),
            "Kosten": round(sum(c for c in (self._row_cost(r) for r in rows) if c is not None), 4),
        }])
        print(f"Servierte Judge-Modellversion(en): {versions or ['-- noch keine --']}")
        return overview

    def estimate(self, output_name="judge_results"):
        """Kostenschätzung ohne API-Aufrufe (Zeichen/4-Heuristik + Ist-Hochrechnung)."""
        answers = self.load_answers()
        prompt_chars = sum(
            len(self.system_prompt) + len(self.build_user_prompt(row)) for row in answers
        )
        estimated_input = prompt_chars // 4
        est_input_cost = estimated_input / 1e6 * self.prices["input"]
        print(f"Bewertungen ausstehend gesamt: {len(answers)} Aufrufe")
        print(
            f"Geschätzte Input-Tokens (Zeichen/4): ~{estimated_input:,} "
            f"(~{est_input_cost:.2f} $ Input-Kosten)"
        )

        rows = self._read_rows(self._output_files(output_name, dry_run=False)[0])
        result = {
            "calls_total": len(answers),
            "estimated_input_tokens": estimated_input,
        }
        completion_values = [self._to_int(r.get("Completion_Tokens")) for r in rows]
        completion_values = [v for v in completion_values if v is not None]
        if completion_values:
            avg_completion = sum(completion_values) / len(completion_values)
            projected_output = avg_completion * len(answers)
            projected_cost = est_input_cost + projected_output / 1e6 * self.prices["output"]
            print(
                f"Ist-Wert aus {len(completion_values)} Urteilen: ~{avg_completion:,.0f} "
                f"Output-Tokens je Bewertung; Hochrechnung Gesamtkosten: ~{projected_cost:.2f} $"
            )
            result["projected_total_cost"] = projected_cost
        else:
            print(
                "Output-Tokens: vor den ersten Echt-Bewertungen nicht prognostizierbar "
                "(Umfang der Zerlegung erst im Ist-Verbrauch sichtbar). "
                f"Grobe Orientierung: bei ~2.000 Output-Tokens je Urteil "
                f"~{est_input_cost + len(answers) * 2000 / 1e6 * self.prices['output']:.2f} $ gesamt."
            )
        return result

    # ------------------------------------------------------------------
    # Datenzugriff
    # ------------------------------------------------------------------

    def load_answers(self):
        """Lädt alle Experiment-Antworten aus den Echtlauf-CSVs (stabil sortiert)."""
        answers = []
        for filename in sorted(os.listdir(self.results_dir)):
            if (
                filename.startswith("experiment_results_run")
                and filename.endswith(".csv")
                and "_DRYRUN" not in filename
            ):
                answers.extend(self._read_rows(os.path.join(self.results_dir, filename)))
        if not answers:
            raise FileNotFoundError(
                f"Keine Experiment-Ergebnisse in {self.results_dir} gefunden."
            )
        answers.sort(key=self._condition_key)
        return answers

    def build_user_prompt(self, answer_row):
        """Rendert den kombinierten Bewertungsprompt für eine Experiment-Antwort."""
        article_id = answer_row["Article_ID"]
        variant_file = f"{article_id}_{answer_row['Method']}.html"
        article = self.loader.load_processed(variant_file)
        key_messages = self.key_messages[article_id]["kernaussagen"]
        numbered = "\n".join(f"{i}. {km}" for i, km in enumerate(key_messages, start=1))
        return self.user_prompt_template.format(
            article=article, key_messages=numbered, answer=answer_row["Answer"]
        )

    def _km_count(self, article_id):
        return len(self.key_messages[article_id]["kernaussagen"])

    # ------------------------------------------------------------------
    # Interne Helfer (Muster aus ExperimentRunner)
    # ------------------------------------------------------------------

    def _base_row(self, answer_row):
        return {
            "Timestamp": datetime.now().isoformat(),
            "Article_ID": answer_row["Article_ID"],
            "Article_Type": answer_row.get("Article_Type", ""),
            "Method": answer_row["Method"],
            "Seed": answer_row["Seed"],
            "Model": answer_row["Model"],
            "Judge_Model": getattr(self.connector, "model_name", ""),
            "Judge_Model_Version": "",
            "Claims_Korrekt": "",
            "Claims_Intrinsisch": "",
            "Claims_Extrinsisch": "",
            "Claims_Gesamt": "",
            "KM_Covered": "",
            "KM_Gesamt": "",
            "IA": "",
            "HR": "",
            "KMC": "",
            "Judge_JSON": "",
            "Prompt_Tokens": "",
            "Completion_Tokens": "",
            "Total_Tokens": "",
        }

    @staticmethod
    def _condition_key(row):
        return (row["Article_ID"], row["Method"], row["Model"], str(row["Seed"]))

    @staticmethod
    def _normalize_key(key):
        article_id, method, model, seed = key
        return (str(article_id), str(method), str(model), str(seed))

    def _output_files(self, output_name, dry_run):
        suffix = "_DRYRUN" if dry_run else ""
        return (
            os.path.join(self.results_dir, f"{output_name}{suffix}.csv"),
            os.path.join(self.results_dir, f"{output_name}_errors{suffix}.csv"),
        )

    def _read_done_keys(self, results_file):
        return {self._condition_key(row) for row in self._read_rows(results_file)}

    @staticmethod
    def _read_rows(file_path):
        if not os.path.exists(file_path):
            return []
        with open(file_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=";"))

    @staticmethod
    def _append_csv_row(file_path, columns, row):
        write_header = not os.path.exists(file_path)
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";", extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _to_int(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _row_cost(self, row):
        prompt_tokens = self._to_int(row.get("Prompt_Tokens"))
        completion_tokens = self._to_int(row.get("Completion_Tokens"))
        if prompt_tokens is None or completion_tokens is None:
            return None
        return (
            prompt_tokens / 1e6 * self.prices["input"]
            + completion_tokens / 1e6 * self.prices["output"]
        )

    def _print_session_costs(self, rows):
        prompt_sum = sum(self._to_int(r.get("Prompt_Tokens")) or 0 for r in rows)
        completion_sum = sum(self._to_int(r.get("Completion_Tokens")) or 0 for r in rows)
        costs = [c for c in (self._row_cost(r) for r in rows) if c is not None]
        summary = (
            f"Token-Verbrauch dieser Sitzung: {prompt_sum:,} Input + "
            f"{completion_sum:,} Output"
        )
        if costs:
            summary += f" (~{sum(costs):.4f} $ laut Preisliste)"
        print(summary)
