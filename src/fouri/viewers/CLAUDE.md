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
  dropdown) and before its window closes, the base frame asks Save, Discard or Cancel when `_unsaved_rounds`
  names any, naming them; a round switch asks nothing. After Save or Discard on closing, the editor closes the
  window itself, so napari's own close question doesn't follow (napari's Ctrl+W and Ctrl+Q still ask theirs
  first). `alignment.AlignmentViewer` is 02's optional check and 03: it starts each
  round from its saved transform or from none, clicks snap to cell centers, Fit (Shift+F) needs 3 pairs with
  two clicks a third of the image apart. A failed well or round switch leaves the old one on screen and puts
  the reason in the prompt; a round the viewer doesn't offer (the anchor, or one the well lacks) opens its first
  round instead, with the reason in the prompt. `segmentation.SegmentationViewer` checks and corrects the
  segmentation masks of every round, opening on the anchor: each of the round's channels is an image layer in
  magma (`R2 DNA`), DNA shown to start and the signals shown kept across round and well switches, under the
  round's masks drawn as cyan and green outlines (`R2 masks`, one Labels layer per round, each with its own Undo,
  emptied when a well opens), each channel's contrast opening at its 1st and 99.5th percentiles. In the pan tool
  (the default) or the pick tool a click selects a cell and Shift+click adds or takes one out, drawn white in the
  `selected cells` layer; a drag pans, and Escape or a click on the background clears the selection (in the
  paint, fill and polygon tools Escape stays napari's own). Delete deletes the selected cells, Shift+M merges them into the first one's
  number when they form one group touching at a side or a corner, Shift+N paints a new cell with the next free
  number, and the brush and eraser widen and trim cells. Save writes every round edited through `io.write_tiff`,
  first keeping each one's Cellpose original. The panel is docked as its Qt widget (`panel.native`): napari
  resets a docked magicgui panel's dropdowns whenever a layer is added, removed or renamed, which would empty
  the round list. 03 no longer uses Affinder, whose 0.5.0 calls `viewer._canvas_size`, removed in napari 0.9.
- `python -m fouri.viewers alignment --config <experiment.json> --well B02 [--round R2]` opens one viewer as
  its own program (the names are `EDITORS` in `viewers/__main__.py`: `alignment` and `segmentation`) and waits until its window closes, as the
  app's Review page will start it: exit code 0 when the window is closed, 2 for an unknown editor or a well not
  in the experiment, 1 when the viewer can't be opened (no experiment.json, a well whose files won't load).
