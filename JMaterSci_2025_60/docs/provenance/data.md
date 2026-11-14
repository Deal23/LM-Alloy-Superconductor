# Data provenance and limitations

The canonical inventory is [data/README.md](../../data/README.md); a machine-readable
inventory is [data_inventory.csv](data_inventory.csv). Current checksums are in
[artifact_manifest.csv](artifact_manifest.csv); see the correction below.
The original manual filtering and file notes, with a Chinese reading guide, are
preserved in [records/](records/README.md).

## Tc correction: Ca3Ir3.64Co0.36Sn13

The original MDR export and earlier repository snapshots recorded **75 K**, while
the local manually curated source recorded **7.5 K**. At the author's request,
`source/mdr.csv` and `processed/mdr_clean.csv`
now record 7.5 K. This is a correction to the working datasets, not independent
verification of the publication or a regeneration of the paper's results.
The original MDR export remains unchanged. Historical prediction tables retain
75 K, and the `od2`/`od5` augmentation snapshots contain 60 K for this formula;
these snapshots and trained models were not regenerated and must not be treated
as results obtained from the corrected dataset. Earlier file versions remain in
Git history at commit `82898a6`.

## Sources

- NIMS MDR/SuperCon: https://doi.org/10.48505/nims.3739
- Source landing page recorded by the author:
  https://mdr.nims.go.jp/concern/datasets/m900p111d?locale=en

Negative-control data used to test whether the models predict near-zero Tc for
known non-superconductors is stored in [`data/non-superconductors/`](../../data/non-superconductors/).
`non.csv` holds 386 candidate non-superconductors with a per-row literature
citation and `Type=Available_0`. Two literature-derived lists accompany it:
`konno_2021_training_removed.csv` (257 entries; the filename indicates training-set duplicates were removed) and
`hosono_2015_formulas.txt` (formula list only). The underlying publications are
*Science and Technology of Advanced Materials* 16, 033503 (2015) and
*Physical Review B* 103, 014509 (2021); only derived lists are redistributed, and
the original third-party PDF is not, per the restriction below.

The raw MDR export is GB18030-encoded (not UTF-8). Other curated CSVs use UTF-8.
`Tc` is in kelvin. The paper data includes oxides, hydrides and other systems.

## Lineage

The historical `mdr_clean_wt.csv` was byte-identical to `source/mdr.csv` and is
represented by that source file rather than stored as a second copy.

```text
source/20240322_mdr_oandm.csv   33,851 rows
       | manual/literature filtering and corrections (not fully automated)
source/mdr.csv                20,995 rows
       | notebook formula standardization and sum≈100 conversion heuristic
processed/mdr_clean.csv       20,995 rows
       | experiments, retained as frozen snapshots
augmented/mdr_clean_{od,os}{2,5}.csv
```

The original cleaning notes contain arithmetic/count inconsistencies. Counts in
this document are measured from files, not inferred from those handwritten notes.
The upstream manual selection cannot be reconstructed by `prepare_data.py` alone.
The current working tables include the correction described above; historical snapshots remain available in Git.

A smoke rebuild has the same 20,995 rows but differs in 101 formula strings and
up to 0.0005 K from the frozen table. Examples include manual Cu-subscript edits
and zero-subscript removal. These differences confirm that the original notebook
alone is not a complete executable record of manual data curation; the processed table alone does not reconstruct every original manual curation step.

## Important restrictions

1. Subscripts summing to approximately 100 are assumed to be mass percentages in
   the original cleaning rule. Atomic percentages can also sum to 100; this is
   not a reliable unit detector.
2. Equal formulas may have distinct Tc measurements/conditions. The default
   cleaning script does not average them; averaging is an explicit option.
3. `os`/`od` are historical filenames. Code, handwritten labels and notebook plot
   legends do not consistently agree on the method-to-label mapping. The saved
   tables and extracted column names are preserved; no labels are silently swapped.
4. SMOTE's standardized-coordinate reconstruction and `ceil(Tc)` behavior are
   documented in [reproducibility.md](../reproducibility.md), not corrected in place.
5. MIT covers this repository's software; upstream datasets, literature and
   published artwork retain their own terms. Users must verify source licenses
   and citation requirements. No publisher PDF, referee correspondence, internal
   approvals or third-party article PDFs were added.
