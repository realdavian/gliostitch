# analysis/

One-off investigations that informed a decision, kept so the decision can be
revisited on the evidence instead of re-derived from scratch.

These are not part of the pipeline. They read datasets directly, are not
imported by `gbm_manifest` or `gbm_os`, and are not covered by the test suite.
Each one records its outcome in its module docstring.

| script | question | outcome |
|---|---|---|
| `lumiere_vital_status.py` | LUMIERE ships a survival time but no event indicator. Can vital status be recovered? | Evidence says yes — no censoring ceiling, follow-up continues past imaging, RANO trajectory ends in progressive disease. Not applied: the manifest stores `os_event = None` and the inference is left to `gbm_os`. |

Run them directly:

```bash
python analysis/lumiere_vital_status.py
```
