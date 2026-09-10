import csv
import math
import os

import pytest

from experiment.analysis import ResultsAnalyzer


# ----------------------------------------------------------------------
# Hilfsfunktionen: synthetische Judge-Ergebnisse schreiben
# ----------------------------------------------------------------------

CSV_HEADER = [
    "Timestamp", "Article_ID", "Article_Type", "Method", "Seed", "Model",
    "Judge_Model", "Judge_Model_Version", "Claims_Korrekt", "Claims_Intrinsisch",
    "Claims_Extrinsisch", "Claims_Gesamt", "KM_Covered", "KM_Gesamt",
    "IA", "HR", "KMC", "Judge_JSON", "Prompt_Tokens", "Completion_Tokens",
    "Total_Tokens",
]


def write_results(path, rows):
    """rows: Liste von Dicts mit Article_ID/Method/Model/Seed/IA/HR/KMC."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in CSV_HEADER}, **row})


def small_analyzer(tmp_path, rows, **overrides):
    """Analyzer auf einem kleinen 2x3x1x2-Design (Artikel x Methode x Modell x Seed)."""
    write_results(tmp_path / "judge_results.csv", rows)
    params = dict(
        results_dir=str(tmp_path),
        articles=("01", "02"),
        methods=("Boost", "Nix", "Original"),
        models=("modell-a",),
        seeds=("1", "2"),
    )
    params.update(overrides)
    return ResultsAnalyzer(**params)


def small_design_rows(ia_fn, hr_fn=None, kmc_fn=None,
                      articles=("01", "02"), methods=("Boost", "Nix", "Original"),
                      models=("modell-a",), seeds=("1", "2")):
    """Erzeugt Zeilen des kleinen Designs; Werte kommen aus fn(article, method, model, seed)."""
    hr_fn = hr_fn or (lambda a, m, mo, s: 5.0)
    kmc_fn = kmc_fn or (lambda a, m, mo, s: 80.0)
    rows = []
    for a in articles:
        for m in methods:
            for mo in models:
                for s in seeds:
                    rows.append({
                        "Article_ID": a, "Method": m, "Model": mo, "Seed": s,
                        "IA": ia_fn(a, m, mo, s), "HR": hr_fn(a, m, mo, s),
                        "KMC": kmc_fn(a, m, mo, s),
                    })
    return rows


# ----------------------------------------------------------------------
# Holm-Korrektur
# ----------------------------------------------------------------------

def test_holm_adjusts_stepdown():
    # sortiert 0.01,0.03,0.04 -> 0.01*3=0.03; 0.03*2=0.06; max(0.06, 0.04*1)=0.06
    assert ResultsAnalyzer._holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_holm_caps_at_one():
    assert ResultsAnalyzer._holm([0.5, 0.9]) == [1.0, 1.0]


def test_holm_preserves_none_and_shrinks_family():
    # None (degenerierte Tests) zaehlen nicht zur Familie: m=2
    assert ResultsAnalyzer._holm([0.02, None, 0.04]) == [0.04, None, 0.04]


def test_holm_empty_and_all_none():
    assert ResultsAnalyzer._holm([]) == []
    assert ResultsAnalyzer._holm([None, None]) == [None, None]


# ----------------------------------------------------------------------
# Matched-Pairs-rangbiseriale Korrelation (Kerby 2014: r = (R+ - R-) / (R+ + R-))
# ----------------------------------------------------------------------

def test_rank_biserial_hand_example():
    # |Diffs| 1,2,3,4 -> Raenge 1,2,3,4; R+=6, R-=4 -> r = 2/10
    assert math.isclose(ResultsAnalyzer._rank_biserial([1, 2, 3, -4]), 0.2)


def test_rank_biserial_excludes_zeros():
    # nur 1 und -2 werden gerankt: r = (1-2)/3
    assert math.isclose(ResultsAnalyzer._rank_biserial([0, 0, 1, -2]), -1 / 3)


def test_rank_biserial_all_zero_is_zero_effect():
    assert ResultsAnalyzer._rank_biserial([0, 0, 0]) == 0.0


def test_rank_biserial_perfect_effect():
    assert ResultsAnalyzer._rank_biserial([5, 3, 8]) == 1.0


def test_rank_biserial_ties_use_midranks():
    # |2| und |-2| teilen sich Rang 1.5 -> r = 0
    assert ResultsAnalyzer._rank_biserial([2, -2]) == 0.0


# ----------------------------------------------------------------------
# Wilcoxon-Vorzeichen-Rang-Test (zweiseitig, exakt ohne Bindungen)
# ----------------------------------------------------------------------

def test_wilcoxon_exact_all_positive():
    # n=3, alle positiv: exakte zweiseitige p = 2/2^3
    assert math.isclose(ResultsAnalyzer._wilcoxon([1, 2, 3]), 0.25)


def test_wilcoxon_drops_zeros_before_testing():
    # Nulldifferenzen werden vorab entfernt (Wilcoxon-Originalverfahren),
    # danach bleibt der Test exakt.
    assert math.isclose(ResultsAnalyzer._wilcoxon([0, 1, 2, 3]), 0.25)


def test_wilcoxon_degenerate_all_zeros():
    assert ResultsAnalyzer._wilcoxon([0.0, 0.0]) is None


# ----------------------------------------------------------------------
# Vollstaendigkeits-Gate
# ----------------------------------------------------------------------

def test_missing_key_aborts_with_key_list(tmp_path):
    rows = small_design_rows(lambda a, m, mo, s: 90.0)
    removed = rows.pop(0)
    analyzer = small_analyzer(tmp_path, rows)
    with pytest.raises(ValueError) as excinfo:
        analyzer.analyze()
    assert removed["Method"] in str(excinfo.value)
    assert "11 von 12" in str(excinfo.value) or "fehlen" in str(excinfo.value)


def test_duplicate_key_aborts(tmp_path):
    rows = small_design_rows(lambda a, m, mo, s: 90.0)
    rows.append(rows[0])
    analyzer = small_analyzer(tmp_path, rows)
    with pytest.raises(ValueError):
        analyzer.analyze()


def test_unexpected_key_aborts(tmp_path):
    rows = small_design_rows(lambda a, m, mo, s: 90.0)
    rows.append({**rows[0], "Article_ID": "99"})
    analyzer = small_analyzer(tmp_path, rows)
    with pytest.raises(ValueError):
        analyzer.analyze()


# ----------------------------------------------------------------------
# RQ1: Paarung, Effektrichtung, Degenerat-Regel, Holm-Familie
# ----------------------------------------------------------------------

def find_rq1(results, metric, method, model=None):
    for row in results["rq1"]:
        if row["metric"] == metric and row["method"] == method and (
                model is None or row["model"] == model):
            return row
    raise AssertionError(f"RQ1-Zeile nicht gefunden: {metric}/{method}/{model}")


def rq1_fixture(tmp_path):
    # Boost hebt IA je (Artikel, Seed) konstant um 10; Nix laesst alles unveraendert.
    base_ia = {("01", "1"): 90.0, ("01", "2"): 91.0, ("02", "1"): 85.0, ("02", "2"): 88.0}

    def ia(a, m, mo, s):
        return base_ia[(a, s)] + (10.0 if m == "Boost" else 0.0)

    return small_analyzer(tmp_path, small_design_rows(ia)).analyze()


def test_rq1_paired_differences_against_baseline(tmp_path):
    results = rq1_fixture(tmp_path)
    row = find_rq1(results, "IA", "Boost")
    assert row["n_pairs"] == 4
    assert row["n_eff"] == 4
    assert row["median"] == 10.0
    assert row["q25"] == 10.0 and row["q75"] == 10.0
    assert row["r_rb"] == 1.0
    # n=4, alle positiv: exakte zweiseitige p = 2/16
    assert math.isclose(row["p_value"], 0.125)
    assert row["degenerate"] is False


def test_rq1_degenerate_method_reports_null_effect(tmp_path):
    results = rq1_fixture(tmp_path)
    row = find_rq1(results, "IA", "Nix")
    assert row["degenerate"] is True
    assert row["p_value"] is None and row["p_holm"] is None
    assert row["r_rb"] == 0.0
    assert row["significant"] is False
    assert row["n_eff"] == 0


def test_rq1_holm_family_only_counts_computed_tests(tmp_path):
    # Familie IA x modell-a: Nix ist degeneriert -> m=1 -> p_holm == p_value
    results = rq1_fixture(tmp_path)
    row = find_rq1(results, "IA", "Boost")
    assert math.isclose(row["p_holm"], row["p_value"])
    assert row["significant"] is False  # 0.125 > 0.05


def test_rq1_diffs_are_retained_for_figures(tmp_path):
    results = rq1_fixture(tmp_path)
    diffs = results["rq1_diffs"][("IA", "modell-a", "Boost")]
    assert sorted(diffs) == [10.0, 10.0, 10.0, 10.0]


# ----------------------------------------------------------------------
# RQ2: Seed-Mittelung und Spearman-Rangkorrelation
# ----------------------------------------------------------------------

def two_model_analyzer(tmp_path, ia_fn):
    rows = small_design_rows(ia_fn, models=("modell-a", "modell-b"))
    return small_analyzer(tmp_path, rows, models=("modell-a", "modell-b"))


def test_rq2_perfectly_consistent_effects(tmp_path):
    # Effekte je (Methode, Artikel): a: 1..4, b: monotone Transformation (Quadrat).
    # Seed-Rauschen +/-2 hebt sich im Mittel auf -> ohne Mittelung waeren die
    # Werte verzerrt, mit Mittelung ist rho exakt 1.
    effect = {("Boost", "01"): 1.0, ("Boost", "02"): 2.0,
              ("Nix", "01"): 3.0, ("Nix", "02"): 4.0}

    def ia(a, m, mo, s):
        base = 80.0
        noise = 2.0 if s == "1" else -2.0
        if m == "Original":
            return base + noise
        e = effect[(m, a)]
        return base + (e if mo == "modell-a" else e ** 2) + noise

    results = two_model_analyzer(tmp_path, ia).analyze()
    row = next(r for r in results["rq2"] if r["metric"] == "IA")
    assert row["rho"] == 1.0
    assert row["n"] == 4


def test_rq2_opposite_effects(tmp_path):
    effect = {("Boost", "01"): 1.0, ("Boost", "02"): 2.0,
              ("Nix", "01"): 3.0, ("Nix", "02"): 4.0}

    def ia(a, m, mo, s):
        if m == "Original":
            return 80.0
        e = effect[(m, a)]
        return 80.0 + (e if mo == "modell-a" else -e)

    results = two_model_analyzer(tmp_path, ia).analyze()
    row = next(r for r in results["rq2"] if r["metric"] == "IA")
    assert row["rho"] == -1.0


def test_rq2_constant_deltas_yield_none(tmp_path):
    # modell-b reagiert auf keine Methode -> keine Varianz -> rho nicht berechenbar
    def ia(a, m, mo, s):
        if mo == "modell-b" or m == "Original":
            return 80.0
        return 80.0 + (1.0 if m == "Boost" else 2.0)

    results = two_model_analyzer(tmp_path, ia).analyze()
    row = next(r for r in results["rq2"] if r["metric"] == "IA")
    assert row["rho"] is None


# ----------------------------------------------------------------------
# Ende-zu-Ende auf dem echten 240er-Design (synthetische Werte)
# ----------------------------------------------------------------------

def full_synthetic_analyzer(tmp_path):
    analyzer = ResultsAnalyzer(results_dir=str(tmp_path))
    rows = []
    for a in analyzer.articles:
        for mi, m in enumerate(analyzer.methods):
            for moi, mo in enumerate(analyzer.models):
                for si, s in enumerate(analyzer.seeds):
                    rows.append({
                        "Article_ID": a, "Method": m, "Model": mo, "Seed": s,
                        "IA": 90.0 + mi + int(a) + (si - 2) * 0.5,
                        "HR": 5.0 + (mi % 3) + moi + (si % 2) * 0.25,
                        "KMC": 80.0 + 2 * mi - moi + si * 0.1,
                    })
    write_results(tmp_path / "judge_results.csv", rows)
    return analyzer


def test_full_design_dimensions(tmp_path):
    results = full_synthetic_analyzer(tmp_path).analyze()
    assert results["n_units"] == 240
    # 3 Metriken x 2 Modelle x 7 Methoden
    assert len(results["rq1"]) == 42
    assert all(row["n_pairs"] == 15 for row in results["rq1"])
    # 3 Metriken, je n = 21 Bedingungen
    assert len(results["rq2"]) == 3
    assert all(row["n"] == 21 for row in results["rq2"])
    # Deskriptiv: 2 Modelle x 8 Varianten (inkl. Original)
    assert len(results["deskriptiv"]) == 16


def test_full_design_exports(tmp_path):
    analyzer = full_synthetic_analyzer(tmp_path)
    results = analyzer.analyze()

    csv_paths = analyzer.export_csv(results)
    for path in csv_paths:
        assert os.path.exists(path)
    with open(csv_paths[0], newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f, delimiter=";"))) == 42

    tables_dir = tmp_path / "tables"
    tex_paths = analyzer.export_latex(results, str(tables_dir))
    assert {os.path.basename(p) for p in tex_paths} == {
        "rq1_ia.tex", "rq1_hr.tex", "rq1_kmc.tex", "rq2_konsistenz.tex", "deskriptiv.tex",
    }
    content = open(tex_paths[0], encoding="utf-8").read()
    assert "\\toprule" in content and "," in content  # booktabs + Dezimalkomma

    images_dir = tmp_path / "images"
    pdf_paths = analyzer.export_figures(results, str(images_dir))
    # explorativ_umfang.pdf fehlt hier bewusst: das synthetische Design fuellt
    # Claims_Gesamt nicht, daher entfaellt die explorative Zusatzanalyse.
    assert {os.path.basename(p) for p in pdf_paths} == {
        "rq1_effekte_ia.pdf", "rq1_effekte_hr.pdf", "rq1_effekte_kmc.pdf",
        "rq2_konsistenz.pdf", "deckensaettigung.pdf",
    }
    for path in pdf_paths:
        assert os.path.getsize(path) > 1000
