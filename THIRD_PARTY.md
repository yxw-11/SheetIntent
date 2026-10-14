# Third-party assets and release notes

- Qwen3-4B-Thinking-2507 weights/tokenizer: obtain from the original model
  publisher and retain its license. Base weights are not bundled here.
- SpreadsheetBench and Spreadsheet-RL workbooks: download from their official
  releases and follow their dataset licenses. No raw workbooks are bundled.
- Frozen recipe files contain dataset-derived instructions and observations.
  They are not an independently licensed dataset. Review upstream permissions
  and sensitive spreadsheet content before public redistribution.
- verl is fetched at v0.6.0 under its upstream license. The small patch in
  `patches/` changes infrastructure compatibility, not its algorithm.
- The optional official comparator download retains its upstream LICENSE,
  revision and hashes. Its declared license is CC-BY-SA-4.0. It is not silently
  relicensed as original project code.
- This extraction does not assign a new license to the original project's
  code. The authors must choose an appropriate code license before public
  release. Keep all existing upstream notices.

The source map in `provenance/core_sources.json` identifies the originating
research modules. Release-specific changes include path portability, removal
of experiment orchestration, fail-closed sandbox errors, and stale-output
removal between repair attempts. These changes are confined to this directory;
the historical project and experiment results remain untouched.
