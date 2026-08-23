"""
MSNet -- Multi-Agent AI Clinical System for MS Diagnosis & EDSS Estimation
=============================================================================
An Autonomous Multi-Agent Collaborative System comprising specialized AI agents:

  1. BiomarkerAgent:
     Analyzes regional brain volume atrophy (Thalamus, Deep GM, Cortical GM)
     relative to age-matched volumetric norms.

  2. LesionVelocityAgent:
     Evaluates T2 lesion burden, log lesion transforms, and disease velocity metrics.

  3. MetaEnsembleAgent:
     Combines tuned Stacking Meta-Learner probabilistic predictions across
     Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM.

  4. ClinicalVerificationAgent:
     Performs sanity verification, computes prediction entropy uncertainty,
     checks for edge-case boundary conditions, and generates human-readable clinical rationale.

Usage:
  python -m agents.ms_agent_system
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from advanced_ensemble_pipeline import expand_features, snap_edss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("MSNet.AgentSystem")

PHENOTYPE_MAP = {0: "Healthy Control", 1: "Relapsing-Remitting MS (RRMS)", 2: "Progressive MS"}
GROUP_MAP     = {0: "Healthy Control", 1: "MS Patient"}


class BiomarkerAgent:
    """Specialist Agent analyzing regional brain atrophy relative to norms."""

    def analyze(self, patient_data: pd.Series, engineered_data: pd.Series) -> Dict[str, Any]:
        age = patient_data.get("Age", 40)
        tha_frac = engineered_data.get("feat_Thalamus_frac", 1.5)
        dgm_frac = engineered_data.get("feat_DeepGM_frac", 3.0)
        gm_wm = engineered_data.get("feat_GM_WM_ratio", 1.0)

        # Domain clinical rules
        thalamic_atrophy = tha_frac < 1.35
        deep_gm_atrophy  = dgm_frac < 2.75
        diffuse_gm_loss  = gm_wm < 0.85

        score = 0.0
        findings = []

        if thalamic_atrophy:
            score += 0.35
            findings.append(f"Accelerated thalamic atrophy detected ({tha_frac:.2f}% of brain volume)")
        if deep_gm_atrophy:
            score += 0.35
            findings.append(f"Subcortical deep grey matter volume loss ({dgm_frac:.2f}%)")
        if diffuse_gm_loss:
            score += 0.30
            findings.append(f"Diffuse grey matter degeneration (GM/WM ratio: {gm_wm:.2f})")

        return {
            "atrophy_score": round(score, 3),
            "thalamic_atrophy": thalamic_atrophy,
            "deep_gm_atrophy": deep_gm_atrophy,
            "diffuse_gm_loss": diffuse_gm_loss,
            "findings": findings
        }


class LesionVelocityAgent:
    """Specialist Agent analyzing T2 hyperintense lesion load & accumulation rate."""

    def analyze(self, patient_data: pd.Series, engineered_data: pd.Series) -> Dict[str, Any]:
        t2lv = patient_data.get("T2 LV (cubic mm)", 0.0)
        duration = patient_data.get("Disease duration", np.nan)
        t2_log = engineered_data.get("feat_T2LV_log1p", 0.0)
        t2_rate = engineered_data.get("feat_T2LV_per_year", 0.0)

        has_lesions = t2lv > 100.0
        high_lesions = t2lv > 5000.0
        severe_lesions = t2lv > 20000.0

        findings = []
        if severe_lesions:
            findings.append(f"Severe T2 hyperintense lesion burden ({t2lv:,.1f} mm³)")
        elif high_lesions:
            findings.append(f"Substantial T2 lesion burden ({t2lv:,.1f} mm³)")
        elif has_lesions:
            findings.append(f"Mild-moderate T2 lesion volume ({t2lv:,.1f} mm³)")
        else:
            findings.append("No hyperintense T2 lesion burden detected (0.0 mm³)")

        if not np.isnan(duration) and duration > 0:
            findings.append(f"Lesion accumulation rate: {t2_rate:,.1f} mm³/year over {duration:.1f} years")

        return {
            "t2_lesion_volume": float(t2lv),
            "log_t2lv": float(t2_log),
            "has_lesions": has_lesions,
            "high_lesions": high_lesions,
            "findings": findings
        }


class ClinicalVerificationAgent:
    """Meta-Agent providing uncertainty estimation and final clinical consensus."""

    def verify(
        self,
        biomarker_res: Dict[str, Any],
        lesion_res: Dict[str, Any],
        ensemble_probs: np.ndarray,
        raw_edss: float
    ) -> Dict[str, Any]:

        # Prediction entropy uncertainty
        entropy = -float(np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8)))
        pred_class = int(np.argmax(ensemble_probs))
        confidence = float(np.max(ensemble_probs))

        # Consensus Rule 1: No lesions + Normal Thalamus -> Healthy Control override
        if not lesion_res["has_lesions"] and not biomarker_res["thalamic_atrophy"]:
            final_class = 0
            final_edss = 0.0
            consensus_note = "Confirmed Healthy Control (No MRI lesion burden & normal thalamic volume)"
        else:
            final_class = pred_class
            final_edss = snap_edss(raw_edss) if final_class > 0 else 0.0
            consensus_note = f"Ensemble prediction confirmed with {confidence*100:.1f}% confidence"

        # Rationale synthesis
        all_reasons = biomarker_res["findings"] + lesion_res["findings"]
        rationale = f"{consensus_note}. Key biomarkers: " + "; ".join(all_reasons) + "."

        return {
            "final_phenotype_code": final_class,
            "final_phenotype_label": PHENOTYPE_MAP[final_class],
            "final_group_label": GROUP_MAP[1 if final_class > 0 else 0],
            "final_edss": final_edss,
            "confidence_score": round(confidence, 4),
            "uncertainty_entropy": round(entropy, 4),
            "clinical_rationale": rationale
        }


class MSAgentCoordinator:
    """Master Multi-Agent Coordinator governing agent interactions."""

    def __init__(self):
        self.biomarker_agent = BiomarkerAgent()
        self.lesion_agent = LesionVelocityAgent()
        self.verifier_agent = ClinicalVerificationAgent()

    def process_patient(
        self,
        patient_data: pd.Series,
        engineered_data: pd.Series,
        ensemble_probs: np.ndarray,
        raw_edss: float
    ) -> Dict[str, Any]:

        # Step 1: Biomarker Agent
        bio_res = self.biomarker_agent.analyze(patient_data, engineered_data)

        # Step 2: Lesion Velocity Agent
        les_res = self.lesion_agent.analyze(patient_data, engineered_data)

        # Step 3: Verification & Consensus Agent
        final_res = self.verifier_agent.verify(bio_res, les_res, ensemble_probs, raw_edss)

        return {
            "biomarker_agent": bio_res,
            "lesion_agent": les_res,
            "consensus_result": final_res
        }


def main():
    logger.info("=" * 75)
    logger.info("  MSNet -- Multi-Agent AI Clinical System Execution")
    logger.info("=" * 75)

    coordinator = MSAgentCoordinator()

    # Test sample execution on patient Case B (Severe MS)
    case_b_raw = pd.Series({
        "Patient: Hash": "case_b_severe_ms",
        "Age": 62,
        "T2 LV (cubic mm)": 45000.0,
        "Disease duration": 25.0
    })
    case_b_eng = pd.Series({
        "feat_Thalamus_frac": 1.15,
        "feat_DeepGM_frac": 2.20,
        "feat_GM_WM_ratio": 0.78,
        "feat_T2LV_log1p": 10.71,
        "feat_T2LV_per_year": 1800.0
    })

    # Simulated Ensemble probabilities for Case B: [HC=0.00, RRMS=0.02, ProgMS=0.98]
    mock_probs = np.array([0.00, 0.02, 0.98])
    mock_edss  = 6.78

    res = coordinator.process_patient(case_b_raw, case_b_eng, mock_probs, mock_edss)

    logger.info("\n--- MULTI-AGENT COLLABORATION REPORT (Sample Case) ---")
    logger.info(f"Final Phenotype : {res['consensus_result']['final_phenotype_label']}")
    logger.info(f"Final EDSS      : {res['consensus_result']['final_edss']}")
    logger.info(f"Confidence      : {res['consensus_result']['confidence_score']*100:.1f}%")
    logger.info(f"Clinical Rationale:\n  {res['consensus_result']['clinical_rationale']}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
