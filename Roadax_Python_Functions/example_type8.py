"""
Example script to run a Type 8 design using the unified facade.

Runs with built-in sample values and prints key outputs.
"""
from unified_api import example_type8_args, run_feature


def main():
    # Get sample args and run the Type 8 design via the facade
    args = example_type8_args()
    result = run_feature("design_by_type", args)

    Best, ResultsTable, Shared, TRACE, T, hFig = result

    print("\n=== Type 8 Design (Example) ===")
    if Best is None or Best.empty:
        print("No feasible design.")
    else:
        print("Best thicknesses (mm): BT=%.0f, Base=%.0f, Subbase=%.0f" % (
            Best.loc[Best.index[0], "BT"],
            Best.loc[Best.index[0], "Base"],
            Best.loc[Best.index[0], "Subbase"],
        ))
        if "CRL" in Best.columns:
            print("CRL=%.0f" % Best.loc[Best.index[0], "CRL"])
        print("Cost (₹ lakh/km):", float(Best.loc[Best.index[0], "Cost"]))

    # Show the derived names and moduli in Shared
    print("\nLayer names:")
    print(" - Base:", Shared.get("Layer_Names", [None, None, "Base", "Subbase"][2]))
    print(" - Subbase:", Shared.get("Layer_Names", [None, None, "Base", "Subbase"][3]))

    # Optionally save the result table to CSV
    # ResultsTable.to_csv("unified_design_result.csv", index=False)


if __name__ == "__main__":
    main()
