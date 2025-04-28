# Usage

```bash
uv run deidentify.py sample/identified/*.svs \
    --salt "super-secret-pepper" \
    -o sample/deidentified \
    -m sample/hash_mapping.csv
```
