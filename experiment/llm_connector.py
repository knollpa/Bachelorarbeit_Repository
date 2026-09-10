import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()

class LLMConnector:
    """
    Kapselt die Kommunikation mit OpenAI (ChatGPT) und Google (Gemini).
    Ermöglicht den einfachen Wechsel zwischen Modellen für den wissenschaftlichen Vergleich.

    Die Modellversionen sind fixiert und werden je Ergebniszeile protokolliert.
    Sampling-Parameter (Temperature, Top-p) werden bewusst nicht gesetzt, damit die
    anbieterseitigen Standardwerte gelten (Abschnitt "Sampling-Parameter und Seed"
    der Thesis); die stochastische Streuung wird stattdessen über den Seed kontrolliert.
    """

    # Fixierte Modellversionen (vgl. Abschnitt "Modellauswahl" der Thesis).
    # OpenAI: datierter Snapshot von GPT-5.4 mini. Google: Stable-ID, die laut
    # Versionsschema des Anbieters auf einen festen Modellstand verweist
    # (models.list meldet dafür 3.5-flash-05-2026); eine separat datierte ID
    # existiert nicht. Die je Antwort gemeldete konkrete Modellversion wird
    # zusätzlich zurückgegeben und im Ergebnisdatensatz protokolliert.
    MODEL_IDS = {
        "chatgpt": "gpt-5.4-mini-2026-03-17",
        "gemini": "gemini-3.5-flash",
    }

    def __init__(self, provider="chatgpt", model_name=None):
        self.provider = provider.lower()
        if self.provider not in self.MODEL_IDS:
            raise ValueError(
                f"Provider '{provider}' wird nicht unterstützt. Verfügbar: {sorted(self.MODEL_IDS)}"
            )
        self.model_name = model_name or self.MODEL_IDS[self.provider]

        # API-Keys aus den Umgebungsvariablen laden
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")

        # Initialisierung des jeweiligen Providers
        if self.provider == "chatgpt":
            if not self.openai_key:
                raise ValueError("OPENAI_API_KEY fehlt!")
            self.client = OpenAI(api_key=self.openai_key)

        elif self.provider == "gemini":
            if not self.gemini_key:
                raise ValueError("GOOGLE_API_KEY fehlt!")
            self.client = genai.Client(api_key=self.gemini_key)

    def send_query(self, system_prompt, user_prompt, seed=None, max_retries=3, backoff_seconds=2.0):
        """
        Sendet eine Anfrage an das konfigurierte Modell.

        Rückgabe ist ein Dict mit der Rohantwort, dem anbieterseitig gemeldeten
        Token-Verbrauch (Grundlage der Kostenkontrolle) und der tatsächlich
        servierten Modellversion (Nachweis der Versionsfixierung):
            {"text": str,
             "prompt_tokens": int | None,
             "completion_tokens": int | None,
             "total_tokens": int | None,
             "model_version": str | None}

        Fehlgeschlagene Aufrufe werden mit exponentiellem Backoff wiederholt; nach
        max_retries Fehlversuchen wird eine Exception ausgelöst, damit Fehlermeldungen
        nie als Modellantwort in den Ergebnisdatensatz gelangen.
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "chatgpt":
                    return self._call_openai(system_prompt, user_prompt, seed)
                return self._call_gemini(system_prompt, user_prompt, seed)
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
            f"{self.model_name}: Anfrage nach {max_retries} Versuchen fehlgeschlagen: {last_error}"
        ) from last_error

    def _call_openai(self, system_prompt, user_prompt, seed=None):
        # Chat-Completions-Endpunkt, da die Responses-API keinen seed-Parameter anbietet.
        request_args = {"seed": seed} if seed is not None else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **request_args,
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("Leere Antwort von OpenAI erhalten.")

        # completion_tokens enthält bei Reasoning-Modellen auch die internen
        # Reasoning-Tokens - genau deshalb wird der Ist-Verbrauch protokolliert.
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "text": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_version": getattr(response, "model", None),
        }

    def _call_gemini(self, system_prompt, user_prompt, seed=None):
        config_args = {"system_instruction": system_prompt}
        if seed is not None:
            config_args["seed"] = seed
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(**config_args),
            contents=user_prompt,
        )
        if not response.text:
            raise RuntimeError("Leere Antwort von Gemini erhalten.")

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        # Output als total - prompt, damit auch Thinking-Tokens enthalten sind,
        # die Google wie Ausgabe-Tokens abrechnet.
        if total_tokens is not None and prompt_tokens is not None:
            completion_tokens = total_tokens - prompt_tokens
        else:
            completion_tokens = getattr(usage, "candidates_token_count", None)

        return {
            "text": response.text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_version": getattr(response, "model_version", None),
        }
