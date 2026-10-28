"""CI smoke test: the model loads, predicts, and the Streamlit app runs end to end."""
import json
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

root = Path(__file__).parent
model = joblib.load(root / "model" / "ebm.joblib")
meta = json.load(open(root / "model" / "meta.json"))

row = {c: (meta["categorical"][c][0] if c in meta["categorical"] else meta["numeric"][c][2]) for c in meta["columns"]}
p = float(model.predict_proba(pd.DataFrame([row])[meta["columns"]])[0, 1])
assert 0.0 <= p <= 1.0, p
print(f"model OK, P(High) = {p:.3f}")

at = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
assert len(at.metric) == 2
print("app OK")
