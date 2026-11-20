# fouri.viewers

Read with the root `CLAUDE.md` when a file under `src/fouri/viewers/` is read.

- Hand-work viewers live in `fouri/viewers/`. `base.BaseViewer` (abstract) opens napari with a docked panel:
  well and round dropdowns, a ▲▼◀▶ pad, the viewer's own tools, Undo and Save, and a shortcut list; it binds
  the keys in `KeyboardShortcuts` (Up/Down round, Left/Right well, Shift for first/last, Ctrl+Z, Shift+S;
  napari 0.9 already uses Ctrl+S and most bare letters) and shows prompts in the canvas text overlay.
  Subclasses fill hooks (`_build_layers`, `_tool_widgets`, `_round_choices`, `_load_well`, `_load_round`,
  `save`, `undo`, `_extra_keys`), and may fill `_extra_shortcuts` (lines added to the shortcut list),
  `_start_round` (the round to open on when none is asked for; the first round offered by default) and
  `_unsaved_rounds` (the rounds with unsaved edits; none by default). Before a well switch (arrows, pad or
  dropdown) and before its window closes, the base frame asks Save, Discard or Cancel when `_unsaved_rounds` names any,
  naming them; a round switch shows no question, except in a viewer with `asks_on_round_switch` set (the alignment
  viewer). After Save or Discard on closing, the editor closes the window itself, so napari's own close question
  doesn't follow (napari's Ctrl+W and Ctrl+Q still ask theirs first). When Save is chosen and the save raises, the
  rounds stay unsaved and the switch or the close does not happen: the window stays open, with the reason in the
  prompt (`Not saved: <reason>. The window stays open.` on closing). A well or round switch that raises any error while
  loading leaves the old one on screen, picker and editor both, and puts the reason in the prompt.
  `alignment.AlignmentViewer` checks Align rounds'
  transforms and aligns rounds by hand: it starts each round from its saved transform or from none (a round beyond the
  saved transforms file, added after Align rounds ran, has none), clicks snap to cell
  centers, Fit (Shift+F) needs 3 pairs with two clicks a third of the image apart. A Fit made since the round's last
  Save or load and not saved counts the round shown as unsaved (`unsaved_fit`), so a well switch, a round switch, a
  change of Start from and closing the window show the Save, Discard or Cancel question; Cancel keeps the Fit (and puts
  Start from back), and Discard drops the Fit. Save of a round beyond the file adds unaligned rows up to it
  (`align.save_manual_transform`). A round the viewer
  doesn't offer (the anchor, or one the well lacks) opens its first round instead, with the reason in the prompt.
  `segmentation.SegmentationViewer` checks and corrects the
  segmentation masks of every round, opening on the anchor: each of the round's channels is an image layer in
  magma (`R2 DNA`), DNA shown to start and the signals shown kept across round and well switches, under the
  round's masks drawn as cyan and green outlines (`R2 masks`, one Labels layer per round, each with its own Undo,
  emptied when a well opens), each channel's contrast opening at its 1st and 99.5th percentiles. In the pan tool
  (the default) or the pick tool a click selects a cell and Shift+click adds or takes one out, drawn white in the
  `selected cells` layer; a drag pans, and Escape or a click on the background clears the selection (in the
  paint, fill and polygon tools Escape stays napari's own). Delete deletes the selected cells, Shift+M merges them into the first one's
  number when they form one group touching at a side or a corner, Shift+N paints a new cell with the next free
  number, and the brush and eraser widen and trim cells. An undo (Ctrl+Z) or a redo (Ctrl+Shift+Z, both bound by the
  editor on each round's masks layer) marks the round as edited, after a Save included,
  so Save writes the round again and leaving shows the question. Save writes every round edited through
  `io.write_tiff`, first keeping each one's Cellpose original; when the write fails, a Cellpose original copied by
  that Save is removed and the error is shown. `artifacts.ArtifactViewer` is the editor for painting artifact masks by
  hand on every round of the well, opening on the anchor: per channel of the round shown, an image layer in magma (`R2 CyclinD1`),
  the painted artifacts (`R2 CyclinD1 artifacts`, one number per artifact; one Labels layer and one Undo history per
  channel of each round, emptied when a well opens) and the final mask drawn as a 2 px outline (`R2 CyclinD1 grown`).
  A channel's three layers show and hide together; the alignment channel is shown to start and the signals shown are
  kept across switches. The edited channel is the channel with the active layer in the layer list; a channel made the
  edited one, by selecting one of its layers or by a switch (always the alignment channel), has its layers shown.
  Click, Shift+click, drag and Escape select as in the segmentation editor, Delete deletes the selected artifacts and
  Shift+N paints a new artifact with the next free number. Spread (0–6) grows an artifact into the touching
  4-connected regions of pixels more than 7 − spread noise widths above the local background
  (`artifacts.excess_over_background`, `artifacts.grow_artifacts`; 0 grows no region) and Pad (0–20 px) dilates the
  grown artifact; both set the selected artifacts, or with
  none selected every artifact of the round shown in every channel. Each artifact keeps its own spread and pad, and a
  newly painted one starts from its channel's values in the `.json`. Minimum mark size (0–200 px, 25 at every opening,
  never saved) excludes smaller painted artifacts from the final mask. The line under the sliders counts the anchor's
  cells left after `clear_border` with a nucleus or ring (Measure cells' default width and gap) touching, once moved
  into a round, that round's final mask in any channel: the round shown as grown now, the other rounds as saved. Without a
  transform for a round, anchor labels, or usable masks files in another round, the line gives the reason in place of
  the count. Save first runs any regrow still queued from the last stroke, so the final mask holds it, then writes
  every round edited: the `.json` with `edited` true, each channel's spread and pad as its file
  had them and each artifact's under `artifacts`, then the `.npz` with `outlines_` and `final_` per channel, both
  through `io.written_whole`; when the `.npz` write fails, the round's previous `.json` is written back, or the new
  one removed when the round had none, and the error is shown. Find artifacts never replaces such a round. A round with an `.npz` lacking outlines for a
  channel or holding an array of another shape, or with an unreadable `.json`, is refused and never written over. The
  panel is docked as its Qt widget (`panel.native`): napari
  resets a docked magicgui panel's dropdowns whenever a layer is added, removed or renamed, which would empty
  the round list. The hand alignment no longer uses Affinder, whose 0.5.0 calls `viewer._canvas_size`, removed
  in napari 0.9.
- `python -m fouri.viewers alignment --config <experiment.json> --well B02 [--round R2]` opens one viewer as
  its own program (the names are `EDITORS` in `viewers/__main__.py`: `alignment`, `segmentation` and `artifacts`) and waits until its window closes, as the
  app's Review page will start it: exit code 0 when the window is closed, 2 for an unknown editor or a well not
  in the experiment, 1 when the viewer can't be opened (no experiment.json, a well whose files won't load).
