from scripts.run_conversations import check


def test_check_passes_on_facts_and_citation():
    r = check(("C1", 1), "You get 18 days [employee_handbook_v3.md].", {"employee_handbook_v3.md"})
    assert r["facts_ok"] and r["cites_retrieved"]


def test_superseded_value_as_the_answer_fails():
    r = check(("C1", 1), "You get 15 days and maybe 18 [leave_policy_v2.1_2024_SUPERSEDED.md].",
              {"leave_policy_v2.1_2024_SUPERSEDED.md"})
    assert not r["facts_ok"] and r["forbidden"]


def test_superseded_value_marked_outdated_is_fine():
    # The real first-run answer, which the old check wrongly failed.
    answer = ("You receive 18 days of annual leave per year [employee_handbook_v3.md]. Note that the previous "
              "policy stating 15 days is outdated and has been superseded [leave_policy_v2.1_2024_SUPERSEDED.md].")
    r = check(("C1", 1), answer, {"employee_handbook_v3.md"})
    assert r["facts_ok"]


def test_check_flags_missing_fact_and_unretrieved_citation():
    r = check(("C1", 2), "Up to 5 days [made_up.md].", {"employee_handbook_v3.md"})
    assert r["missing"] == ["31 March"]
    assert not r["cites_retrieved"]
