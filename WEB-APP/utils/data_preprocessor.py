import pandas as pd
from .formula_parser import parse_formula


def calculate_atomic_ratio(composition, element_mass):
    total_atom = sum(
        fraction / element_mass[element] for element, fraction in composition.items()
    )
    return {
        element: (fraction / element_mass[element]) / total_atom
        for element, fraction in composition.items()
    }


def preprocess_molecular_data(df):
    all_elements = set()
    for formula in df["formula"]:
        all_elements.update(parse_formula(formula).keys())

    feature_data = []
    target_data = []

    for index, formula in df["formula"].items():
        elements_ratios = parse_formula(formula)
        feature_row = {
            element: elements_ratios.get(element, 0) for element in all_elements
        }
        feature_data.append(feature_row)
        target_data.append(
            df.loc[index, "Tc"] if not pd.isna(df.loc[index, "Tc"]) else None
        )

    return pd.DataFrame(feature_data), pd.Series(target_data)
