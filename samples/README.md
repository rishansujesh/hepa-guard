# Sample Payloads

`si_units_example.json` uses SI mmol/L values for HDL, glucose, and triglycerides with `meta.units_version="si"`. The API converts those values to mg/dL before validation and inference.

`recommendations_example.json` mirrors the `/predict` request schema and can be used to test `POST /recommendations`:

```bash
curl -s -X POST "http://127.0.0.1:8000/recommendations" \
  -H "Content-Type: application/json" \
  --data @samples/recommendations_example.json | python -m json.tool
```
