"""CLI: erzeugt alle 21 Varianten reproduzierbar.

Aufruf:  cd Python_Bachelorarbeit && python -m transform.cli
Benötigt ANTHROPIC_API_KEY (in .env oder Umgebung).
"""
from __future__ import annotations
import os
from . import variant_generator as vg


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY fehlt (.env oder Umgebung setzen).")

    client = anthropic.Anthropic(api_key=api_key)
    written = vg.generate_all(client)
    print(f"{len(written)} Varianten erzeugt:")
    for p in written:
        print(" -", p.name)


if __name__ == "__main__":
    main()
