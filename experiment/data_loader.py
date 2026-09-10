from bs4 import BeautifulSoup
import os

class DataLoader:
    """
    Verantwortlich für das Einlesen, Vorverarbeiten und Speichern der
    Presseartikel aus dem lokalen Dateisystem.

    Ordnerstruktur:
        data/raw/       -> originale, unveränderte Presseartikel
    """

    CONTENT_SELECTORS = [
        "h1",
        ".div_date_location",
        ".text_summary",
        ".imported-article",
    ]

    TAGS_TO_REMOVE = [
        "script", "style", "header", "footer", "nav",
        "aside", "link", "meta", "noscript", "iframe",
    ]

    SELECTORS_TO_REMOVE = [
        ".pp-images-module",
        "#colorbox",
        "#pp_cboxOverlay",
        ".pp_breadcrumb__container",
        ".pp-search__modal",
        ".div_like_reactions",
        ".media-button__container",
        "#placeholderFBSDK",
    ]

    def __init__(
        self,
        raw_path="data/raw",
        processed_path="data/processed",
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(self.processed_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(self) -> dict:
        """
        Liest die Rohartikel aus data/raw ein, bereinigt sie und speichert
        das Ergebnis in data/processed.

        Namenskonvention der Ausgabedateien:
            raw/01_Original.html       -> processed/01_Original.html

        Es wird ausschließlich data/raw verarbeitet: Die bereits transformierten
        und verifizierten Methodenvarianten in data/processed dürfen von der
        Bereinigung nicht erneut angefasst (und damit überschrieben) werden.

        Rückgabe:
            dict mit Statistiken je verarbeiteter Datei.
        """
        sources = [
            (self.raw_path, "raw"),
        ]

        stats = {}
        for folder, label in sources:
            if not os.path.exists(folder):
                print(f"[WARNUNG] Ordner nicht gefunden, wird übersprungen: {folder}")
                continue

            for filename in os.listdir(folder):
                if not filename.endswith(".html"):
                    continue

                input_path = os.path.join(folder, filename)
                output_path = os.path.join(self.processed_path, filename)

                try:
                    cleaned = self._load_and_clean(input_path)
                    self._save(cleaned, output_path)
                    file_stats = self.get_token_stats(cleaned)
                    file_stats["quelle"] = label
                    stats[filename] = file_stats
                    print(f"[OK] {label}/{filename} -> {file_stats['token_geschaetzt']} Token (geschätzt)")
                except Exception as e:
                    print(f"[FEHLER] {label}/{filename}: {e}")
                    stats[filename] = {"quelle": label, "fehler": str(e)}

        return stats

    def load_processed(self, filename: str) -> str:
        """Lädt eine bereits bereinigte Datei aus data/processed."""
        file_path = os.path.join(self.processed_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bereinigte Datei nicht gefunden: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def list_available_articles(self, folder="raw") -> list[str]:
        """Gibt eine Liste aller HTML-Dateien im angegebenen Ordner zurück."""
        paths = {
            "raw": self.raw_path,
            "processed": self.processed_path,
        }
        path = paths.get(folder, self.raw_path)
        if not os.path.exists(path):
            return []
        return [f for f in os.listdir(path) if f.endswith(".html")]

    def get_token_stats(self, content: str) -> dict:
        """Grobe Token-Schätzung (1 Token ≈ 4 Zeichen)."""
        char_count = len(content)
        return {
            "zeichen": char_count,
            "token_geschaetzt": char_count // 4,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _load_and_clean(self, file_path: str) -> str:
        """Liest eine HTML-Datei ein und gibt den bereinigten Inhalt zurück."""
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self._clean_html(html_content)

    def _clean_html(self, html: str) -> str:
        """
        Extrahiert nur die inhaltlich relevanten Bereiche des Presseartikels.
        Die HTML-Struktur der Inhaltselemente bleibt für die GEO-Untersuchung erhalten.
        """
        soup = BeautifulSoup(html, "html.parser")

        # 1a. JSON-LD vor dem Entfernen der Scripts sichern
        json_ld_strings = [
            str(tag)
            for tag in soup.find_all(
                "script",
                type=lambda value: value
                and value.strip().lower() == "application/ld+json",
            )
        ]

        # 1b. Entferne irrelevante Tags inkl. Inhalt
        for tag in soup(self.TAGS_TO_REMOVE):
            tag.decompose()

        # 2. Entferne spezifische UI-Komponenten per CSS-Selektor
        for selector in self.SELECTORS_TO_REMOVE:
            for element in soup.select(selector):
                element.decompose()

        # 3. Extrahiere ausschließlich die relevanten Inhaltsbereiche
        extracted_parts = []
        for selector in self.CONTENT_SELECTORS:
            for element in soup.select(selector):
                extracted_parts.append(element.decode(formatter="html"))

        if not extracted_parts:
            raise ValueError(
                "Keine relevanten Inhaltsbereiche gefunden. "
                "Prüfe die CSS-Selektoren oder die HTML-Struktur der Datei."
            )

        # 4. Zusammenführen in ein minimales HTML-Dokument
        json_ld_content = "\n".join(json_ld_strings)
        body_content = "\n".join(extracted_parts)
        return f"<article>\n{json_ld_content}\n{body_content}\n</article>"

    def _save(self, content: str, output_path: str) -> None:
        """Speichert bereinigten Inhalt in data/processed."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
