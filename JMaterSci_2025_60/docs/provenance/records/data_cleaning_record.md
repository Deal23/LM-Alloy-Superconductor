# Data-Cleaning Record

Source dataset:
https://mdr.nims.go.jp/concern/datasets/m900p111d?locale=en

## 1. Missing Critical-Temperature Values

Initial records: `33,465`

Records removed because the recommended transition temperature was missing: `7,107`

Remaining records: `26,358`

## 2. Formulas with Unknown Variables or Missing Subscripts

Records before filtering: `26,358`

Removed records:

- Formulas containing `-` or `+`: `3,243`
- Formulas ending with `x`, `y`, `z`, `X`, `Y`, or `Z`: `2,113`

Remaining records: `21,002`

## 3. Unreliable Records

Records before filtering: `21,002`

Removed records:

1. *Physica C: Superconductivity* **351**, 78-81 (2001): `Pb2C1Ag2O6`, reported `310 K`, unverified.
2. *Nature* **244**, 244 (2023): `Lu1H`, reported `294 K`, retracted.
3. *Nature* **586**, 373 (2020): `C1S1`, reported `288 K`, retracted.
4. "Superconductivity found in meteorites": `Mg1C1Na1Sn1Ca1Al1Si1Ni1In1Pb1`; not treated as a decenary alloy record.
5. *Physica C* **246**, 207-215 (1995): `Hg0.66Pb0.34Ba2Ca1.98Cu2.9O8.4`; `143 K` corresponds to a diamagnetic-transition temperature.
6. *AIP Advances* **11**, 015007 (2021): `Y1Ba2Cu3La0.67Sr0.33Mn1O3OO7`; element entry error.

Records without traceable sources: `42`

Remaining records: `20,950`

## 4. High-Pressure Superconductors

Records before filtering: `20,996`

Removed high-pressure records:

1. *Nature* **569**, 528 (2019): `La1H10`, `250 K`.
2. *Nature* **525**, 73 (2015): `H2S1`, `203 K` at `155 GPa`.
3. `cond-mat/1412.0460`: `H2S1`, `185 K` at `143 GPa`.
4. *Physica C* **552**, 27 (2018): `D3S1`, `148 K`.
5. *Physica C* **552**, 27 (2018): `H3S1`, `147 K`.
6. *Nature* **525**, 73-76 (2015): `H2S1`, `135 K` at `200 GPa`.

Remaining records: `20,947`

## 5. Manual Corrections

1. *Physica B* **281**, 909 (2000): `Gd0.6Pr0.4Ba2Cu3O6.6` and `Gd0.7Pr0.3Ba2Cu3O6.5` were corrected to `Y0.6Pr0.4Ba2Cu3O6.6` and `Y0.7Pr0.3Ba2Cu3O6.5` because of element-entry errors.
2. *Physical Review B* **60**, 10489 (1999): `Y1Ba2Cu3O6.1`; reported `90 K` and `92 K` values require Br restoration and were set to `0 K`.
3. *Physical Review B* **98**, 014518 (2018): `Cs1Eu1Fe4As4`; `287 K` was corrected to `28.7 K` because of a data-entry error.

## 6. Invalid Element Symbols

Records before filtering: `20,947`

Invalid-symbol records removed: `339`

Remaining records: `20,608`

Detected invalid symbols:

| Symbol | Count |
| --- | ---: |
| `Ah` | 1 |
| `Bf` | 1 |
| `Ct` | 3 |
| `M`, `D`, `T` | 327 |
| `Dc` | 1 |
| `Gs` | 1 |
| `Jr` | 1 |
| `Ls` | 1 |
| `Pf` | 1 |
| `Ph` | 1 |
| `Rr` | 1 |
