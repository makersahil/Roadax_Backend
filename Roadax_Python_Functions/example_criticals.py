"""
Example script to run Critical Strain Analysis using the unified facade.
"""
from unified_api import example_criticals_args, run_feature


def main():
    args = example_criticals_args()
    out = run_feature("critical_strain_analysis", args)
    # out is a 4x1 vector: [Critical_BSi, Critical_CTBSi, CFD, Critical_SubgradeSi]
    print("Critical Strains (microstrain) and CFD:")
    labels = [
        "Bituminous Critical (microstrain)",
        "CTB Critical (microstrain)",
        "CFD (-)",
        "Subgrade Critical (microstrain)",
    ]
    for label, val in zip(labels, out):
        print(f"- {label}: {val}")


if __name__ == "__main__":
    main()
