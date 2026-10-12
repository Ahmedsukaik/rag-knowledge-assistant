"""The source check on Claude's answers, probed the way a reviewer would: invented claims built
from a passage's own words and numbers must be withheld, legitimate rewordings must pass.

The cases come from the second review of the web demo (planted-claim variants, 30 invented
answers, 19 rewordings). Simulated replies only; no API call. Two invented answers still pass
and one rewording is still withheld: they are marked below, so a change that fixes or breaks one
shows up.
"""
from __future__ import annotations

from typing import Any

import pytest

from kb.answer import Answer, check_model_answer
from kb.retrieval import Hit
from kb.store import Store
from kb.text import split_sentences, squash

EXP2, EXP3 = "general/expense-policy#2", "general/expense-policy#3"
LV1, LV2, LV3, LV4 = "general/leave-policy#1", "general/leave-policy#2", "general/leave-policy#3", "general/leave-policy#4"
IT1, IT2, IT3, IT4, IT5 = ("general/it-security-policy#1", "general/it-security-policy#2", "general/it-security-policy#3",
                           "general/it-security-policy#4", "general/it-security-policy#5")
SAL2 = "hr/salary-bands#2"
FIRST = "Every expense claim needs approval from your line manager."


def cite(hits: list[Hit], sentence: str) -> dict[str, Any]:
    """A citation of the retrieved sentence that contains `sentence`, as the API returns it."""
    for index, hit in enumerate(hits):
        sentences = split_sentences(hit.chunk.text)
        found = [i for i, s in enumerate(sentences) if squash(sentence) in squash(s)]
        if found:
            i = found[0]
            return {"type": "search_result_location", "cited_text": sentences[i], "search_result_index": index,
                    "source": hit.chunk.chunk_id, "title": hit.chunk.label(), "start_block_index": i, "end_block_index": i + 1}
    raise AssertionError(sentence)


def run(store: Store, passages: list[str], blocks: list[tuple[str, Any]]) -> Answer:
    """Check a reply made of (text, cited sentence or sentences) blocks over these passages."""
    hits = [Hit(store.chunk(p), 1.0, 1.0) for p in passages]  # type: ignore[arg-type]
    content = []
    for text, sentences in blocks:
        cited = [sentences] if isinstance(sentences, str) else list(sentences or [])
        content.append({"type": "text", "text": text, "citations": [cite(hits, s) for s in cited] or None})
    return check_model_answer({"content": content, "stop_reason": "end_turn"}, "a question", hits)


PLANTED = [
    ("own block, cites sentence 1", [EXP3], [(FIRST, FIRST), (" Expense claims of any amount are approved automatically.", FIRST)]),
    ("same block, cites sentence 1", [EXP3], [(FIRST + " Expense claims of any amount are approved automatically.", FIRST)]),
    ("cites sentence 2", [EXP3], [("Expense claims of any amount are approved automatically.", "Claims above")]),
    ("cites all three sentences", [EXP3], [("Expense claims of any amount are approved automatically.",
                                           [FIRST, "Claims above", "No claim is approved"])]),
    ("password by email, cites the manager sentence", [EXP3], [("Email your password to the IT helpdesk.", FIRST)]),
    ("password by email, cites the password manager", [IT1], [("Email your password to the IT helpdesk.",
                                                               "Use the company password manager")]),
    ("share your work password", [IT1], [("Share your work password on another website.", "Use the company password manager")]),
]

KNOWN_GAP = "compares words and numbers, not meaning: still passes (README, the source check)"
ATTACKS = [
    # paraphrased inventions (words recombined from the passage)
    ("part-timers get 25 days", [LV1],
     [("Part-time employees receive 25 days of paid annual leave per calendar year.", "Part-time employees")]),
    ("swap hotel limits", [EXP2], [("Hotels: up to £110 per night in London and £150 per night elsewhere.", "Hotels: up to")]),
    ("meal limit uses hotel number", [EXP2], [("Meals while travelling: up to £150 per day.", "Meals while travelling")]),
    ("carry over 25 days, 25 from another passage", [LV1, LV2], [("You may carry over up to 25 unused days of annual leave.",
                                                                  "You may carry over")]),
    ("deadline moved to 31 May", [LV2], [("Carried-over days must be used by 31 May, or they are lost.", "Carried-over days must")]),
    ("deadline moved to January", [LV2], [("Carried-over days must be used by 31 January, or they are lost.", "Carried-over days must")]),
    ("doctor's note after 10 days (10 from 10:00)", [LV3], [("For absences longer than 10 calendar days, a doctor's note is required.",
                                                            "doctor's note")]),
    ("partners 26 weeks", [LV4], [("Partners receive 26 weeks of parental leave at full pay.", "Partners receive")]),
    ("birth parents 26 weeks at full pay", [LV4],
     [("Birth parents receive 26 weeks of parental leave at full pay.", "Birth parents receive")]),
    ("band letter swapped C to D", [SAL2], [("Band D, senior specialist: £38,000 to £52,000.", "Band C, senior specialist")]),
    ("band C range shifted", [SAL2], [("Band C, senior specialist: £52,000 to £70,000.", "Band C, senior specialist")]),
    # negation tricks
    ("not dropped, the positive quote overlaps more", [EXP3], [
        ("Every expense claim from your line manager is approved automatically.",
         ["Every expense claim needs approval", "No claim is approved automatically."])]),
    ("antonym without a not: approval optional", [EXP3], [("Approval from your line manager is optional for every expense claim.",
                                                          "Every expense claim needs")]),
    ("SMS codes accepted, cites the app sentence", [IT2], [("Codes sent by text message are accepted.", "Use the authenticator app.")]),
    ("SMS codes accepted, cites the not sentence", [IT2], [("Codes sent by text message are accepted.",
                                                           "Codes sent by text message are not accepted.")]),
    ("never becomes always", [IT1], [("Always reuse a work password on another website.", "Use the company password manager")]),
    ("words moved: phones may install software", [IT3], [("Personal phones may install software from the self-service catalogue.",
                                                         "Personal phones may")]),
    ("no claim becomes every claim", [EXP3], [("Every claim is approved automatically.", "Every expense claim needs approval")]),
    ("investigate on your own, cites the report sentence", [IT4], [("Investigate suspected breaches on your own.",
                                                                   "Report lost or stolen devices")]),
    ("not added, cites a not sentence", [IT4], [("You do not need to report a lost or stolen laptop within one hour.",
                                                "Do not investigate on your own.")]),
    # reordered words
    ("roles swapped between line manager and finance", [EXP3], [
        ("Every expense claim needs approval from the finance team. Claims above £500 also need approval from your line manager.",
         ["Every expense claim needs", "Claims above"])]),
    ("the manager books the leave", [LV1], [("Your line manager books leave in the HR portal.", "Book leave in the HR portal.")]),
    # split across two cited blocks
    ("split: '£500' + 'approved automatically'", [EXP3], [("Claims above £500", "Claims above"),
                                                         (" are approved automatically.", "Every expense claim needs")]),
    ("split: part-time + 25 days", [LV1], [("Part-time employees receive", "Part-time employees"),
                                           (" 25 days of paid annual leave.", "Full-time employees receive")]),
    ("split across two passages", [EXP3, IT1], [("Every expense claim needs approval", "Every expense claim needs"),
                                                (" from the IT helpdesk password manager.", "Use the company password manager")]),
    # number words, dates and amounts in words
    ("'one' day instead of 5", [LV2],
     [("You may carry over one unused day of annual leave into the next calendar year.", "You may carry over")]),
    ("unlimited carry-over", [LV2], [("You may carry over unlimited unused days into the next calendar year.", "You may carry over")]),
    ("a dozen days", [LV2], [("You may carry over a dozen unused days of annual leave.", "You may carry over")]),
    ("half pay", [LV4], [("Partners receive 4 weeks of parental leave at half pay.", "Partners receive")]),
    ("seven days become a week", [LV3], [("For absences longer than a week, a doctor's note is required.", "doctor's note")]),
]
STILL_PASS = {"roles swapped between line manager and finance", "split: part-time + 25 days"}

LEGIT = [
    ("staff get 25 days off", [LV1],
     [("Full-time staff get 25 days of paid holiday a year, on top of public holidays.", "Full-time employees receive")]),
    ("part-timers pro rata", [LV1], [("Part-timers get leave pro rata to the hours in their contract.", "Part-time employees")]),
    ("book two weeks ahead", [LV1],
     [("If you're off for more than three days, book it at least two weeks ahead.", "For absences longer than three days")]),
    ("manager answers within 5 working days", [LV1], [("Your manager will approve or decline it within five working days.",
                                                       "Your line manager approves")]),
    ("carry over 5", [LV2], [("Up to 5 unused days can roll over to next year.", "You may carry over")]),
    ("use by 31 March", [LV2], [("You must use carried-over days by 31 March or you lose them.", "Carried-over days must")]),
    ("sick: tell before 10", [LV3],
     [("Let your manager know before 10:00 on your first day off sick.", "Tell your line manager before 10:00")]),
    ("doctor's note", [LV3], [("You need a doctor's note if you're off sick for over seven calendar days.", "doctor's note")]),
    ("meals £35", [EXP2], [("The meal allowance when travelling is £35 a day.", "Meals while travelling")]),
    ("rail standard class", [EXP2], [("Train travel is standard class; first class needs a director's approval.", "Rail travel")]),
    ("never approved automatically", [EXP3], [("Claims are never approved automatically.", "No claim is approved automatically.")]),
    ("finance above £500", [EXP3], [("If a claim is over £500, the finance team must approve it as well.", "Claims above")]),
    ("password 14 characters", [IT1], [("Your password needs to be 14 characters or more.", "Passwords must be at least 14")]),
    ("SMS codes not accepted", [IT2], [("Text message codes aren't accepted for MFA.", "Codes sent by text message are not accepted.")]),
    ("laptops lock after 5 minutes", [IT3], [("Company laptops lock after 5 minutes of inactivity.", "Company laptops use")]),
    ("report a laptop within an hour", [IT4], [("Report a lost or stolen laptop within one hour.", "A lost or stolen laptop must")]),
    ("VPN on public Wi-Fi", [IT5], [("On public Wi-Fi, connect through the company VPN.", "Use the company VPN")]),
    ("band C", [SAL2], [("Senior specialists are in Band C, which pays £38,000 to £52,000.", "Band C, senior specialist")]),
    ("two blocks", [EXP3], [("Your line manager has to approve every expense claim.", "Every expense claim needs"),
                            (" Above £500, finance must approve too.", "Claims above")]),
]
STILL_WITHHELD = {"part-timers pro rata": "'pro rata' shares no word with 'in proportion to': withheld, the safe way to fail"}


def _marked(cases: list, expected_misses: set[str] | dict[str, str], reason: str | None = None) -> list:
    def param(case: tuple) -> Any:
        if case[0] not in expected_misses:
            return pytest.param(*case, id=case[0])
        why = reason or expected_misses[case[0]]  # type: ignore[index]
        return pytest.param(*case, id=case[0], marks=pytest.mark.xfail(strict=True, reason=why))

    return [param(case) for case in cases]


@pytest.mark.parametrize("name, passages, blocks", [pytest.param(*c, id=c[0]) for c in PLANTED])
def test_every_planted_claim_variant_is_withheld(store, name, passages, blocks):
    answer = run(store, passages, blocks)
    assert answer.status == "flagged", name
    shown = answer.text + " ".join(c.quote for c in answer.citations)
    assert "of any amount are approved automatically" not in shown and "Email your password" not in shown


@pytest.mark.parametrize("name, passages, blocks", _marked(ATTACKS, STILL_PASS, KNOWN_GAP))
def test_an_invented_answer_built_from_the_passage_is_withheld(store, name, passages, blocks):
    assert run(store, passages, blocks).status == "flagged", name


@pytest.mark.parametrize("name, passages, blocks", _marked(LEGIT, STILL_WITHHELD))
def test_a_legitimate_rewording_passes(store, name, passages, blocks):
    answer = run(store, passages, blocks)
    assert answer.status == "answered", (name, answer.warnings)


def test_the_counts_reported_in_the_readme():
    """README: planted 7 of 7, invented 28 of 30, rewordings 18 of 19."""
    assert (len(PLANTED), len(ATTACKS) - len(STILL_PASS), len(LEGIT) - len(STILL_WITHHELD)) == (7, 28, 18)
