# Pretrained artifact manifest

`manifest.json` accepts either one Release file per artifact or an ordered `parts` list.
Every downloaded file and every reassembled archive is checked by exact byte size and
SHA256 before extraction.

Single-file entry:

```json
{
  "name": "Avalon OOF",
  "filename": "avalon_oof_predictions.tar.gz",
  "size": 123,
  "sha256": "<archive-sha256>",
  "url": "<release-url>",
  "oof_only": true
}
```

Split entry (parts must appear in concatenation order):

```json
{
  "name": "Avalon models",
  "filename": "avalon_fingerprint_models.tar.gz",
  "size": 456,
  "sha256": "<combined-archive-sha256>",
  "oof_only": false,
  "parts": [
    {"filename": "avalon_fingerprint_models.tar.gz.part-00", "size": 200, "sha256": "<part-sha256>", "url": "<release-url>"},
    {"filename": "avalon_fingerprint_models.tar.gz.part-01", "size": 256, "sha256": "<part-sha256>", "url": "<release-url>"}
  ]
}
```

The current `manifest.json` records the packaged `v1.0.0-manuscript` assets. Upload all
four archives plus `SHA256SUMS` without renaming them, then test both full and
inference-only downloads from a clean checkout before publishing the release.
