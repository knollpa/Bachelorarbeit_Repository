"""Korrigierte Transformationsprompts (vgl. thesis.tex Anhang, Reference R-01..R-07)."""
from __future__ import annotations
from dataclasses import dataclass

_SYSTEM = (
    'You are an expert editor specializing in press release optimization for AI '
    'systems (Generative Engine Optimization). Your task is to apply the "{name}" '
    "method to a given press article. Return only the optimized article - no "
    "explanations, no meta-commentary."
)

_ARTICLE_BLOCK = "\n\n[ARTICLE START]\n\n{article}\n\n[ARTICLE END]\n\nReturn the optimized article in the same HTML format as the input."


@dataclass(frozen=True)
class Method:
    key: str
    name: str          # human label used in the prompt
    filename: str      # suffix of data/processed/<id>_<filename>.html
    rules: str         # the "Apply ... RULES: ..." body (Reference R-0X)
    needs_key_messages: bool = False


# NOTE: `rules` strings are the canonical Reference R-01..R-07 from the plan.
METHODS: dict[str, Method] = {
    "statistics_addition": Method(
        "statistics_addition", "Statistics Addition", "StatisticsAddition",
        '''Apply the "Statistics Addition" method to the following press article.
RULES:
1. REPLACE vague or qualitative statements with concrete, quantifiable data ONLY where a specific figure is explicitly stated elsewhere in the article or is unambiguously derivable from stated facts.
2. MAKE EXISTING latent numbers explicit (a count, range, date, or measurement already implied by the text). If a qualitative phrase ("significantly", "faster", "leading") has no figure stated or derivable in the article, LEAVE IT UNCHANGED - do NOT invent one.
3. KEEP all existing statistics exactly as they are - do not alter any number already present in the source text.
4. DO NOT invent, estimate, or fabricate figures. Only use data that is (a) explicitly stated in the article or (b) unambiguously derivable from stated facts.
5. PRESERVE the original HTML structure, all tags (<h1>, <p>, <strong>, <blockquote>, etc.), the JSON-LD block, and the full article length.
6. DO NOT rewrite, reorder, or summarize content - only add or substitute numerical specificity where rule 1 applies.
7. DO NOT omit or weaken any key statement of the original article.
8. RESPOND in the same language as the input article.''',
    ),
    "quotation_addition": Method(
        "quotation_addition", "Quotation Addition / Cite Sources", "QuotationAddition",
        '''Apply the "Quotation Addition / Cite Sources" method to the following press article.
RULES:
1. IDENTIFY statements that are ALREADY direct quotes from named persons (executives, spokespersons, experts) and wrap them in HTML <blockquote> tags, preserving the speaker attribution inside or directly after the tag.
2. DO NOT fabricate verbatim wording: only wrap statements already presented as quotations. Do NOT convert paraphrased or indirect speech into invented verbatim quotes.
3. CITE SOURCES: for claims that reference an external study, standard, norm, regulation, or measurement protocol that is named or unambiguously identifiable in the article (e.g., "WLTP", "ISO 12906"), add an explicit, inline source attribution to that named source. Do NOT invent sources or citations not identifiable from the article.
4. KEEP all existing <blockquote> markup and existing source citations in place - do not alter them.
5. PRESERVE all other HTML tags, structure, the JSON-LD block, and content unchanged.
6. DO NOT reorder, add, or remove any content beyond the <blockquote> tagging and the source attributions described above.
7. DO NOT omit or weaken any key statement of the original article.
8. RESPOND in the same language as the input article.''',
    ),
    "fluency_optimization": Method(
        "fluency_optimization", "Fluency Optimization", "FluencyOptimization",
        '''Apply the "Fluency Optimization" method to the following press article.
RULES:
1. SIMPLIFY overly technical jargon by replacing it with plain, widely understood language where possible WITHOUT losing precision. Do not change technical terms that carry a factual specification (model names, units, standards).
2. SPLIT long, complex sentences (more than approximately 30 words) into shorter, clearer sentences.
3. REMOVE redundant words and filler phrases.
4. PRESERVE all factual information exactly - do not alter names, numbers, dates, or technical specifications.
5. MAINTAIN the original HTML structure, all tags, and the JSON-LD block; do not add or omit paragraphs or sections.
6. DO NOT reorder, add, or omit any facts or content - only improve sentence-level clarity and readability.
7. DO NOT change the grammatical voice (active/passive) of sentences; voice is handled by a different method.
8. RESPOND in the same language as the input article.''',
    ),
    "authoritative_tone": Method(
        "authoritative_tone", "Authoritative Tone", "AuthoritativeTone",
        '''Apply the "Authoritative Tone" method to the following press article.
RULES:
1. ATTRIBUTE statements explicitly to the issuing company or institution using active constructions (e.g., "The BMW Group confirms...", "According to the BMW Group...", "The BMW Group announces...").
2. CONVERT passive voice constructions to active voice where the company or a named actor is the subject.
3. PRESERVE the original degree of certainty of every statement. Do NOT convert hedged, conditional, or forward-looking statements (e.g., "might", "could", "is expected to", "plans to") into definitive claims. Authority comes from clear attribution and active voice, not from increasing certainty.
4. PRESERVE all factual content - do not alter names, numbers, dates, or technical specifications.
5. MAINTAIN the original HTML structure, all tags, and the JSON-LD block unchanged.
6. DO NOT invent claims, add new information, or change the meaning or certainty of any statement.
7. DO NOT omit or weaken any key statement of the original article.
8. RESPOND in the same language as the input article.''',
    ),
    "logical_structure": Method(
        "logical_structure", "Logical Structure", "LogicalStructure",
        '''Apply the "Logical Structure" method to the following press article.
RULES:
1. IDENTIFY groups of technical specifications, product features, or comparable factual data currently presented in prose form.
2. CONVERT such groups into HTML <table> elements with appropriate column headers to improve scannability and machine-readability.
3. INTRODUCE or UPGRADE heading tags (<h2>, <h3>) to create a clear, hierarchical structure where thematic sections currently lack explicit headings.
4. PRESERVE all factual content exactly - do not alter numbers, names, dates, or technical specifications.
5. DO NOT delete factual content and do not reorder major content sections; reformatting prose into tables is permitted, but all remaining paragraphs keep their original order.
6. KEEP all existing HTML tags not affected by the restructuring, including the JSON-LD block, unchanged.
7. DO NOT omit or weaken any key statement of the original article.
8. RESPOND in the same language as the input article.''',
    ),
    "conclusion_first": Method(
        "conclusion_first", "Conclusion First", "ConclusionFirst",
        '''Apply the "Conclusion First" method to the following press article.

The pre-annotated key messages of this article are:
{key_messages}

RULES:
1. MOVE all of the pre-annotated key messages listed above into the first paragraph of the article body, following the inverted-pyramid principle. The first paragraph MUST contain every one of them.
2. ENSURE the opening paragraph clearly states, using ONLY information already present in the article: (a) who the actor is, (b) what was announced or achieved, and (c) the most significant stated implication or benefit.
3. PRESERVE all remaining content in its original order after the relocated key messages.
4. DO NOT duplicate content - if a sentence is moved to the beginning, remove it from its original position.
5. DO NOT invent or add any information not present in the article.
6. MAINTAIN the original HTML structure, all tags, and the JSON-LD block unchanged.
7. RESPOND in the same language as the input article.''',
        needs_key_messages=True,
    ),
    "json_ld": Method(
        "json_ld", "JSON-LD Structured Data", "JSON-LD",
        '''Apply the "JSON-LD Structured Data" method to the following press article.
RULES:
1. LOCATE the existing <script type="application/ld+json"> block in the article's HTML.
2. EXPAND all abbreviated model names, product names, and proper nouns to their full, unambiguous form (e.g., "3er" -> "BMW 3 Series", "iX1" -> "BMW iX1").
3. ADD missing technical specifications that are explicitly stated in the article body but absent from the JSON-LD (e.g., engine output, range, dimensions, certifications).
4. ADD contextual attributes to named entities (e.g., full corporate name, location, role) where only abbreviations or shorthand are currently present.
5. DO NOT invent or fabricate data - only use information explicitly stated in the article's HTML body.
6. PRESERVE the original JSON-LD schema type (@type) and all other structured-data fields that do not require expansion. ENSURE the resulting JSON-LD remains syntactically valid and conformant to its original schema.org type.
7. KEEP all HTML outside the <script> block completely unchanged.
8. RESPOND in the same language as the input article.''',
    ),
}


def format_key_messages(key_messages: list[str]) -> str:
    return "\n".join(f"{i}. {m}" for i, m in enumerate(key_messages, start=1))


def build_prompt(method_key: str, article_html: str,
                 key_messages: list[str] | None = None) -> tuple[str, str]:
    method = METHODS[method_key]
    system = _SYSTEM.format(name=method.name)
    rules = method.rules
    if method.needs_key_messages:
        if not key_messages:
            raise ValueError(f"Method '{method_key}' requires key_messages.")
        rules = rules.replace("{key_messages}", format_key_messages(key_messages))
    user = rules + _ARTICLE_BLOCK.format(article=article_html)
    return system, user
