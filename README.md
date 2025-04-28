# Usage

```bash
uv run deidentify.py "sample/identified/*.{svs,tif,tiff}" \
    --salt "super-secret-pepper" \
    -o sample/deidentified \
    -m sample/hash_mapping.csv
```
