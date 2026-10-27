import json
import math

import pytest

from councileval import metrics as M
from councileval import stats as S
from councileval.prepare import sentence_at, split_for
from councileval.schema import Document, EvalItem, ValidationError, load_items, validate_dataset, validate_item
from councileval.scoring import date_match, is_correct, retrieval_status, taxonomy, yes_no_match


def item(**kw):
    base = dict(id="q1", question="Who built it?", source="test", source_id="1", category="extractive",
                answer_type="span", answers=["Ilse Varga"], answerable=True, doc_id="d1",
                evidence=["It was built by Ilse Varga in 1871."], split="heldout", license="CC BY-SA 4.0", metadata={})
    base.update(kw)
    return EvalItem(**base)


DOCS = {"d1": Document("d1", "t", "# T\n\nIt was built by Ilse Varga in 1871. It is long.", "test", "CC BY-SA 4.0")}


# ── dataset validation ────────────────────────────────────────────────

def test_valid_dataset_passes():
    validate_dataset([item(), item(id="q2")], DOCS)


@pytest.mark.parametrize("bad", [
    dict(id=""), dict(question="  "), dict(category="trivia"), dict(answer_type="essay"), dict(split="train"),
    dict(answers=[]), dict(answers=[""]), dict(answers=[3]), dict(answer_type="none"),
    dict(answerable=False),                                   # unanswerable but lists answers
    dict(answerable=False, answers=[], answer_type="span"),   # unanswerable needs answer_type none
    dict(answer_type="number", answers=["twelve-ish"]),
    dict(doc_id=""), dict(license=""),
])
def test_malformed_items_are_rejected(bad):
    with pytest.raises(ValidationError):
        validate_item(item(**bad))


def test_duplicate_ids_missing_documents_and_missing_evidence_fail():
    with pytest.raises(ValidationError, match="duplicate"):
        validate_dataset([item(), item()], DOCS)
    with pytest.raises(ValidationError, match="missing"):
        validate_dataset([item(doc_id="nope")], DOCS)
    with pytest.raises(ValidationError, match="evidence"):
        validate_dataset([item(evidence=["A sentence that is not in the document."])], DOCS)


def test_malformed_jsonl_line_is_reported(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(item().to_json() + "\n{not json\n")
    with pytest.raises(ValidationError, match="line 2"):
        load_items(p)


def test_split_assignment_is_stable_and_seed_free():
    ids = [f"squad:{i}" for i in range(2000)]
    first = [split_for(i) for i in ids]
    assert first == [split_for(i) for i in ids]
    dev = first.count("dev") / len(ids)
    assert 0.17 < dev < 0.23


def test_evidence_window_covers_answer_across_sentence_boundary():
    ctx = "The St. Louis team won. It was founded by J. R. Smith in 1901. Later it moved."
    start = ctx.index("J. R. Smith")
    ev = sentence_at(ctx, start, start + len("J. R. Smith"))
    assert "J. R. Smith" in ev and "1901" in ev
    assert "Later it moved" not in ev and "team won" not in ev, "window must be minimal, not the whole context"


# ── answer metrics ────────────────────────────────────────────────────

def test_normalization_matches_squad_rules():
    assert M.normalize("The  Eiffel-Tower!") == "eiffeltower"
    assert M.normalize("An apple, a pear") == "apple pear"


def test_exact_match_and_f1():
    assert M.exact_match("the Ilse Varga", ["Ilse Varga"]) is True
    assert M.exact_match("Varga", ["Ilse Varga"]) is False
    assert M.token_f1("Ilse", ["Ilse Varga"]) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert M.token_f1("completely wrong", ["Ilse Varga"]) == 0.0
    long = " ".join(["word"] * 40)
    assert M.exact_match(long, ["word"]) is None and M.token_f1(long, ["word"]) is None, "long answers are not EM/F1-scored"


def test_containment_needs_whole_tokens():
    assert M.answer_containment("It was built by Ilse Varga.", ["Ilse Varga"]) is True
    assert M.answer_containment("Varga built it", ["Ilse Varga"]) is False
    assert M.answer_containment("The year 18710 is wrong", ["1871"]) is False


def test_numeric_match():
    assert M.numeric_match("There were 1,234 votes.", ["1234"]) is True
    assert M.numeric_match("Roughly twelve players", ["12"]) is True
    assert M.numeric_match("It was 3.14 metres", ["3.15"]) is True       # within 1%
    assert M.numeric_match("It was 3.5 metres", ["3.15"]) is False
    assert M.numeric_match("13 players", ["12"]) is False                 # integers exact
    assert M.numeric_match("no numbers here", ["12"]) is False
    assert M.parse_number("seven") == 7 and M.parse_number("n/a") is None


def test_abstention_detection():
    assert M.abstained("The document does not mention the founder.")
    assert M.abstained("This cannot be determined from the provided text.")
    assert not M.abstained("The founder was Ilse Varga.")


def test_type_specific_correctness():
    assert yes_no_match("Yes, both were American.", ["yes"]) is True
    assert yes_no_match("No. They differ.", ["yes"]) is False
    assert date_match("It happened on 3 March 1990.", {"day": "3", "month": "March", "year": "1990"}) is True
    assert date_match("It happened in March 1991.", {"day": "", "month": "March", "year": "1990"}) is False
    un = item(id="u", answers=[], answerable=False, answer_type="none", evidence=[], category="unanswerable")
    assert is_correct(un, "The text does not say.") is True
    assert is_correct(un, "It was Ilse Varga.") is False
    assert is_correct(item(), "It does not mention it, but probably Ilse Varga") is False, "hedged abstention is not correct"


def test_taxonomy_separates_retrieval_and_generation_failures():
    it = item()
    got = retrieval_status(it, [{"content": "It was built by Ilse Varga in 1871."}])
    miss = retrieval_status(it, [{"content": "Unrelated text."}])
    assert taxonomy(it, "Bob built it", False, got) == "wrong_despite_retrieval"
    assert taxonomy(it, "Bob built it", False, miss) == "retrieval_miss"
    assert taxonomy(it, "The document does not say", False, got) == "false_abstention"
    un = item(id="u", answers=[], answerable=False, answer_type="none", evidence=[], category="unanswerable")
    assert taxonomy(un, "Bob", False, retrieval_status(un, [])) == "answered_unanswerable"


# ── ranking metrics ───────────────────────────────────────────────────

def test_ranking_metrics():
    ranked, rel = [4, 2, 9, 1, 7], {9}
    assert M.recall_at_k(ranked, rel, 1) == 0.0 and M.recall_at_k(ranked, rel, 3) == 1.0
    assert M.recall_at_k(ranked, rel, 2) == 0.0, "relevant item at rank 3 is not in the top 2"
    assert M.mrr_at_k(ranked, rel, 5) == pytest.approx(1 / 3)
    assert M.ndcg_at_k(ranked, rel, 5) == pytest.approx(1 / math.log2(4))
    assert M.ndcg_at_k([9, 4], {9}, 5) == 1.0
    assert M.all_recall_at_k(ranked, {2, 1}, 3) == 0.0 and M.all_recall_at_k(ranked, {2, 1}, 4) == 1.0
    assert M.recall_at_k(ranked, set(), 3) is None, "no labels -> not scored, not zero"


def test_reranking_comparison_is_paired_and_detects_change():
    from councileval.retrieval_eval import all_evidence_at_k, auroc
    assert all_evidence_at_k([1, 2, 3], [{1}, {3}], 3) == 1.0
    assert all_evidence_at_k([1, 2, 3], [{1}, {5}], 3) == 0.0
    assert auroc([3, 4], [1, 2]) == 1.0 and auroc([1], [1]) == 0.5
    vec = [0.0, 0.0, 1.0, 0.0]
    rr = [1.0, 1.0, 1.0, 0.0]
    d = S.paired_bootstrap_diff(vec, rr)
    assert d["diff"] == pytest.approx(0.5) and d["n"] == 4


# ── confusion matrix and calibration ──────────────────────────────────

def test_confusion_matrix():
    c = M.confusion([True, True, False, False, False], [True, False, True, False, False])
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 2)
    assert c["tpr"] == 0.5 and c["fpr"] == pytest.approx(1 / 3) and c["precision"] == 0.5
    with pytest.raises(ValueError):
        M.confusion([True], [True, False])


def test_calibration_ignores_missing_confidence():
    assert M.brier_score([None, None], [True, False]) is None
    assert M.brier_score([1.0, 0.0, None], [True, False, True]) == 0.0
    assert M.brier_score([0.9], [False]) == pytest.approx(0.81)
    assert M.expected_calibration_error([0.95, 0.95], [True, False]) == pytest.approx(0.45)


# ── statistics ────────────────────────────────────────────────────────

def test_bootstrap_is_reproducible_with_seed():
    a, b = [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 0]
    assert S.paired_bootstrap_diff(a, b) == S.paired_bootstrap_diff(a, b)
    assert S.paired_bootstrap_diff(a, b, seed=1)["ci"] != S.paired_bootstrap_diff(a, b, seed=2)["ci"]


def test_mcnemar_and_wilson():
    r = S.mcnemar_exact([True] * 10 + [False] * 10, [True] * 10 + [True] * 10)
    assert r["b_only"] == 10 and r["a_only"] == 0 and r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert S.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    lo, hi = S.wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2, "0/20 is not 'proven zero'"
    assert S.wilson_interval(0, 0) == (None, None)


def test_verdicts_never_overclaim():
    assert S.describe_difference({"n": 5, "diff": 0.2, "ci": (-0.1, 0.5)}).startswith("no reliable difference")
    assert S.describe_difference({"n": 50, "diff": 0.2, "ci": (0.05, 0.3)}).startswith("improvement")
    assert S.describe_difference({"n": 0, "diff": None, "ci": (None, None)}) == "not measured"
