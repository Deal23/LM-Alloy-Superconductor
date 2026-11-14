# Submission and supplementary material inventory

Only numerical tables, a supplement file index, and embedded supplementary
figures were selected from author-supplied submission materials. Manuscripts and
correspondence are not added to the repository.

## Supplementary CSVs

All **15 submitted supplementary CSVs** were compared with the
existing frozen repository tables using parsed CSV records. Every file matches;
line-ending differences do not change the records. No duplicate CSV copies were
added. [supplementary_inventory.csv](supplementary_inventory.csv) records file
numbers, current repository paths, row counts, submitted-file hashes and the
original Word document's figure/table descriptions.

`Supplementary information.docx` contains one 15-file index table and two embedded
images. Images are preserved without re-rendering:

| Repository image | Original caption |
|---|---|
| `paper/published/figures/supplement_binary_et.png` | Fig. 8, predictions for 2,145 binary alloys, ET model |
| `paper/published/figures/supplement_binary_rf.png` | Fig. 9, predictions for 2,145 binary alloys, RF model |

These are extracted author submission assets, not a claim of verification against
the publisher's final online supplementary files. The Word document's introduction
also refers to Figures 6–7; its displayed captions say Figures 8–9. Both references
are historical inconsistencies, not silently renumbered here.

## Revised manuscript tables

Five tables from the author's revised manuscript
are extracted to `paper/published/tables/`. Cell text, blank merged-cell
continuations, dashes, and ellipsis rows are retained; these are source tables,
not reconstructed measurements or newly calculated metrics.

| File | Contents |
|---|---|
| `table_01.csv` | Data problems and handling methods |
| `table_02.csv` | Literature model comparison, including cited works' reported metrics |
| `table_03.csv` | Hyperparameter search ranges |
| `table_04.csv` | Random Forest hyperparameter results, as ordered in revised manuscript |
| `table_05.csv` | Extra-Trees hyperparameter results, including RMSE 6.2624 K / R² 0.9519 |

The supplementary index associates `hyperparameters_et.csv` with Table 4 and `hyperparameters_rf.csv`
with Table 5. That association is reversed relative to the revised manuscript's
table content. The index and extracted manuscript tables preserve their source
labels; do not use the index alone to identify the estimator. Table 2 includes
third-party literature comparisons, so its values must retain their source
context and must not be interpreted as evaluation on one common dataset.

Sources and checksums for extracted assets are recorded in `artifact_manifest.csv`.
Original source documents are not distributed. No reviewer reports,
suggested-reviewer lists, cover letters, or internal approval records are published.
