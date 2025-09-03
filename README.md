# CRT Scanner – FAST + Robust
- Batch Yahoo downloads + individual retries
- GPW alias map for common mismatches (DIN->DNP, BUD->BDX, AMX->AMC, CIGAMES->CIG, LIVE->LVC, PEK->PBX)
- Short history in 'Okazje C3' mode
- Weekly W1 chart with Trigger/SL/TP/Key overlays

## Run
pip install -r requirements.txt
streamlit run app.py

## Tests
pip install -r requirements-dev.txt
pytest -q
