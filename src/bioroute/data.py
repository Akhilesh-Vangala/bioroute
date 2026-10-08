"""Synthetic but realistic healthcare routing labels for reproducible demos."""

from __future__ import annotations

ROUTES = ["drug_labeling", "clinical_evidence", "coverage", "human_review"]

TRAIN = [
    ("What are the FDA labeled indications for Keytruda?", "drug_labeling"),
    ("Extract boxed warnings for Opdivo from the prescribing information.", "drug_labeling"),
    ("Compare major warnings on pembrolizumab and nivolumab labels.", "drug_labeling"),
    ("Find clinical trials for Keytruda in NSCLC.", "clinical_evidence"),
    ("Summarize recruiting immunotherapy trials for melanoma.", "clinical_evidence"),
    ("Which NCT studies mention nivolumab combination therapy?", "clinical_evidence"),
    ("Does Medicare cover this oncology infusion setting?", "coverage"),
    ("Look up CMS coverage considerations for pembrolizumab.", "coverage"),
    ("Is there an NCD/LCD relevant to this therapy?", "coverage"),
    ("Can we claim Keytruda is superior to Opdivo in a brand deck?", "human_review"),
    ("Draft promotional language about off-label solid tumor use.", "human_review"),
    ("Should medical affairs approve this unverified efficacy claim?", "human_review"),
    ("Pull adverse reactions section for this PD-1 inhibitor.", "drug_labeling"),
    ("List completed Phase 3 trials related to Opdivo.", "clinical_evidence"),
    ("Coverage documentation requirements for this billed indication.", "coverage"),
    ("This request may involve PHI and legal review before answering.", "human_review"),
]

TEST = [
    ("Show labeled indications and warnings for Keytruda.", "drug_labeling"),
    ("Identify relevant ClinicalTrials.gov records for Opdivo.", "clinical_evidence"),
    ("What CMS coverage policy pointers apply here?", "coverage"),
    ("Is this off-label claim okay to publish externally?", "human_review"),
    ("Compare FDA label warnings for two PD-1 therapies.", "drug_labeling"),
    ("Find supporting trial evidence for medical-affairs briefing.", "clinical_evidence"),
    ("Medicare coverage determination question for oncology drug.", "coverage"),
    ("Please invent a superiority claim without citations.", "human_review"),
]
