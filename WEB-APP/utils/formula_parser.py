import re
import numpy as np


def parse_formula(formula):
    """
    Parse chemical formula to element-ratio dictionary.

    Args:
        formula: Chemical formula string (e.g., "Ba0.5In0.5", "YBa2Cu3O7")

    Returns:
        dict: Element symbols as keys, atomic ratios as values
    """
    if formula is None or (isinstance(formula, str) and formula.strip() == ""):
        return {}
    formula = str(formula)
    elements_ratios = {}
    for element, ratio in re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula):
        if element:
            elements_ratios[element] = float(ratio) if ratio else 1.0
    return elements_ratios


def standardize_formula(formula):
    composition = parse_formula(formula)
    result = []
    for element, amount in composition.items():
        if amount == int(amount):
            if amount != 1:
                result.append(f"{element}{int(amount)}")
            else:
                result.append(element)
        else:
            formatted = f"{amount:.3f}".rstrip("0").rstrip(".")
            result.append(f"{element}{formatted}")
    return "".join(result)


def generate_random_formula(elements, num_formulas):
    formulas = []
    while len(formulas) < num_formulas:
        coefficients = np.random.rand(len(elements))
        coefficients /= coefficients.sum()
        coefficients = np.round(coefficients, 3)
        if np.all(coefficients >= 0) and np.isclose(coefficients.sum(), 1):
            formula = "".join(
                f"{elements[i]}{coefficients[i]:.3f}"
                for i in range(len(elements))
                if coefficients[i] != 0
            )
            formulas.append(formula)
    return formulas
