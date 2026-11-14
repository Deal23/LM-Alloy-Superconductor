"""Composition scope shared with LM-based-SC's frozen modeling dataset."""
import re

ELEMENTS = frozenset((
    "Ag Al Am As Au B Ba Be Bi Br C Ca Cd Ce Cl Co Cr Cs Cu Dy Er Eu F Fe Ga Gd Ge "
    "Hf Hg Ho I In Ir K La Li Lu Mg Mn Mo N Na Nb Nd Ni Np Os P Pa Pb Pd Pr Pt Pu "
    "Rb Re Rh Ru S Sb Sc Se Si Sm Sn Sr Ta Tb Tc Te Th Ti Tl Tm U V W Y Yb Zn Zr"
).split())
NONMETALS = frozenset("B C N F Si P S Cl Ge As Se Br Sb Te I".split())
METALS = ELEMENTS - NONMETALS
TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?")
ALL_ELEMENTS = frozenset((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split())


def parse_scoped_formula(formula, allow_zero=False, unrestricted=False):
    text = str(formula).strip()
    amounts = {}
    pos = 0
    for match in TOKEN.finditer(text):
        if match.start() != pos:
            raise ValueError("化学式格式无效，请使用元素符号和正数配比")
        element, number = match.groups()
        if not unrestricted and element in {"O", "H"}:
            raise ValueError("当前体系不包含 O 或 H")
        if element not in (ALL_ELEMENTS if unrestricted else ELEMENTS):
            raise ValueError(f"元素 {element} 不在当前 81 元素建模范围内")
        value = float(number) if number else 1.0
        if value == 0 and allow_zero:
            pos = match.end()
            continue
        if value <= 0:
            raise ValueError("元素配比必须大于零")
        amounts[element] = amounts.get(element, 0.0) + value
        pos = match.end()
    if not amounts or pos != len(text):
        raise ValueError("化学式格式无效，请使用元素符号和正数配比")
    if not unrestricted and not METALS.intersection(amounts):
        raise ValueError("当前体系至少需要包含一种金属元素")
    total = sum(amounts.values())
    return {element: value / total for element, value in amounts.items()}
