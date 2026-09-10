# Cyber Trust by Design: An AI-Driven Governance Model for Digital Resilience

Reproduction repository for the paper "Cyber Trust by Design: An AI-Driven Governance Model for Digital Resilience" (2026).

## Paper Summary

AI governance is fragmented. Five major standards exist — NIST AI RMF, ISO/IEC 42001, MITRE ATLAS, OWASP GenAI, EU AI Act — each covering part of the problem. None covers the whole, and none produces quantitative trust scores.

This paper presents a three-layer governance architecture (Risk and Ethics, Technical Trust, Organizational Assurance) that maps these standards into one metric system. Six trust dimensions — transparency, fairness, robustness, privacy, explainability, accountability — aggregate into a composite score **T**, extended with a four-factor resilience model.

**Key results:**
- Proposed framework: **T = 0.76 ± 0.03**
- NIST-only baseline: T = 0.62
- OWASP-only baseline: T = 0.58
- Improvement is statistically significant (paired t-test, p < 0.01, Cohen's d > 2.0)
- Sector overlays for BFSI, healthcare, and energy tune weights to domain priorities
- Agentic AI extension addresses tool authorization and memory integrity

Cross-framework integration beats any single standard applied alone.

## Quick Start

```bash
pip install -r requirements.txt
python src/evaluation.py
```

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   └── download.py          # Fetch datasets automatically
├── src/
│   ├── trust_model.py       # Core trust scoring (Equation 1)
│   ├── metrics.py           # MEI, BS, ARS, PS, GS computation
│   ├── sectors.py           # BFSI/healthcare/energy overlays
│   └── evaluation.py        # Run all experiments
├── results/
│   ├── adult_results.csv
│   ├── german_results.csv
│   ├── compas_results.csv
│   ├── mimic_results.csv
│   └── fraud_results.csv
└── notebooks/
    └── reproduce.ipynb      # Interactive reproduction
```

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| Adult Income | UCI ML Repository | Bias, fairness, explainability |
| German Credit | UCI ML Repository | Risk scoring, compliance |
| COMPAS | ProPublica | Algorithmic bias, accountability |
| MIMIC-III | PhysioNet (requires credentialed access) | Clinical explainability, safety |
| IEEE Fraud | Kaggle | Adversarial resilience, robustness |

## Citation

```bibtex
@article{damor2026cyber,
  title={Cyber Trust by Design: An AI-Driven Governance Model for Digital Resilience},
  author={Damor, Prabhakar and Kumar, Naveen},
  year={2026}
}
```

## License

MIT
