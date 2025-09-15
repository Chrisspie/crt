# CRT Scanner – FAST + HTF(touch)
- HTF confluence: closest HTF level (Open/Close/Low/High) with tolerance (touch=1%, strict=0.3%+direction)
- Default 'Reguła kontaktu' = touch
- Batch Yahoo downloads + individual retries
- GPW alias map (DIN->DNP, BUD->BDX, AMX->AMC, CIGAMES->CIG, LIVE->LVC, PEK->PBX)

## Run
pip install -r requirements.txt
streamlit run app.py

## Tests
pip install -r requirements-dev.txt
pytest -q
