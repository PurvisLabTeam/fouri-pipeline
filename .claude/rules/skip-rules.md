---
paths:
  - "src/fouri/{preprocess,segment,align,artifacts,measure,status,run}.py"
  - "src/fouri/app/{gpu,steps}.py"
  - "src/fouri/app/pages/status.py"
  - "tests/test_{preprocess,segment,align,measure,run}.py"
---

# Skip rules of the steps

Loaded with the root `CLAUDE.md` when a file matching `paths` is read. Convert images, Segment nuclei, Align
rounds, Find artifacts and Measure cells only create outputs that are missing unless `overwrite` is `True` (root
`CLAUDE.md`, "Files passed between steps"). Per step:

- Convert images skips a well whose `round_table_{well}.csv` (not the old `.pkl`) lists the same rounds. If the
  saved table lists only the first of them, `convert_well` writes just the new rounds' TIFFs and saves the full
  table. If a new round's images are smaller, the crop shrinks: every round is written again and, with or without
  Overwrite, the well's labels in `segmented`, their Cellpose originals in `cellpose_original/` and every array of its
  masks in `masks/{well}` (`io.mask_paths`; the `.json` unchanged) are cut to the new size. The crop only trims the
  bottom and right edges, so outlines and saved transforms keep their positions (within 0.04 px on B02). A changed
  well (root `CLAUDE.md`, under the files table) is refused alone, Overwrite on or off, naming its changed rounds and
  the two ways to confirm (the Run page's question, `python -m fouri.run reconvert`), unless `reconvert_changed` is
  true.
- Segment nuclei skips a round whose labels exist, and never replaces a hand-edited round (one with a file in
  `cellpose_original/`), even with `overwrite=True`; it names those rounds apart from the ones skipped as done.
  The Run page's count of images to segment (`fouri.app.gpu.images_to_segment`) leaves them out too. Right before a
  round's labels are written, the round's Cellpose original is looked for again; a round with one, saved by hand
  while the step ran, is passed over and named once the work ends, apart from the rounds found hand-edited at the
  start.
- Align rounds skips a round whose matrix isn't NaN, so rounds fixed by hand alignment are kept; a saved file
  with fewer rounds keeps them and aligns only the new ones. It refuses a well whose label images don't match its
  rounds, that has no label image for `anchor_round`, or whose saved file has more rounds than the well or a
  non-identity matrix at `anchor_round` (the anchor changed); the checks run in that order and a well gets the first
  one's message. Just before writing, the saved file is read again and only the rounds this run computed are written
  over the file; a round whose saved matrix changed while the step ran (a hand alignment) is passed over and named
  once the work ends, with or without Overwrite.
- Find artifacts skips a round whose `.npz` exists, and never replaces a round whose `.json` has `edited` true
  (masks painted by hand), even with `overwrite=True`; the step names those rounds apart from the ones skipped as
  done. The `.json` is written before the `.npz`, so a round with an `.npz` always has its `.json`. Find artifacts
  refuses a well whose round table is missing or predates `raw_file_name`, or whose round has a `.json` without
  an `edited` of true or false. Right before a round is written, the round's `.json` is read again; a round now
  `edited` is passed over and named once the work ends.
- A refused well is a failed well: it is left out of the work and passed to `check_results` with the others, so
  the other wells still run and the step ends failed naming it.
- Measure cells skips a well whose CSV has a `{prefix}_nuc_mean` column for every channel in its round table; a
  well missing some measures all its rounds again.

The checks run in the main process before work is handed out; `report.say_skipped` prints what was skipped.
`fouri.status` (`step_status` per well, `experiment_status` for Build cell table, Batch effects and Embedding)
applies the same rules, read from disk, for the app's Status page; change it with a step's skip rule.
`fouri.status.measured_wells` gives the wells of `well_list` whose Measure cells status is complete, then the others,
each in plate order; Build cell table and the cell pages (`fouri.app.cells`) take their wells from it.
