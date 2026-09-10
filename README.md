# Cyber Trust by Design: An AI-Driven Governance Model for Digital Resilience

Reproduction repository for the paper "Cyber Trust by Design: An AI-Driven Governance Model for Digital Resilience" (2026).

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
