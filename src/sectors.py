"""
Sector-specific configuration for BFSI, healthcare, and energy.
"""
from .trust_model import apply_sector_overlay, DEFAULT_WEIGHTS


SECTORS = {
    "bfsi": {
        "name": "Banking, Financial Services & Insurance",
        "focus": ["explainability", "fraud_integrity", "hallucination_mitigation"],
        "regulations": ["RBI", "DPDP", "EU AI Act", "PSD2", "FFIEC"],
        "trust_weights": apply_sector_overlay(DEFAULT_WEIGHTS, "bfsi"),
    },
    "healthcare": {
        "name": "Healthcare",
        "focus": ["clinical_safety", "consent_traceability", "bias_mitigation"],
        "regulations": ["NDHM", "DPDP", "MDR", "GDPR", "HIPAA"],
        "trust_weights": apply_sector_overlay(DEFAULT_WEIGHTS, "healthcare"),
    },
    "energy": {
        "name": "Energy & Utilities",
        "focus": ["grid_resilience", "predictive_maintenance", "carbon_optimization"],
        "regulations": ["CEA", "MoP", "ENTSO-E", "DOE", "FERC"],
        "trust_weights": apply_sector_overlay(DEFAULT_WEIGHTS, "energy"),
    },
}


def get_sector_config(sector):
    """Get configuration for a sector."""
    return SECTORS.get(sector.lower(), None)


def list_sectors():
    """List available sectors."""
    return list(SECTORS.keys())
