# Chess AI - Email Notifications

Training now sends epoch summaries (loss/accuracy/correlation) plus attachments (`training_curves.png`, `qualitative_log.txt`) to the configured recipients.

## Configure SMTP credentials

Two options:
1) **.env file (preferred for local/dev)**  
   Create `.env` in the repo root with:
   ```
   SMTP_HOST=smtp.example.com
   SMTP_PORT=587
   SMTP_USER=you@example.com
   SMTP_PASS=your-password-or-app-password
   SMTP_FROM=you@example.com   # optional; defaults to SMTP_USER
   ```
   The code auto-loads `.env` if `python-dotenv` is installed (already handled in `cnn.py`).

2) **Environment variables (good for GCP/prod)**  
   Export before running:
   ```bash
   export SMTP_HOST=...
   export SMTP_PORT=587
   export SMTP_USER=...
   export SMTP_PASS=...
   export SMTP_FROM=...    # optional
   ```
   Or prefix a command:
   ```bash
   SMTP_HOST=... SMTP_PORT=587 SMTP_USER=... SMTP_PASS=... python game_engine/cnn.py
   ```

## Recipients

Hardcoded recipients (per requirements): `adithya@usc.edu`, `krishmod@usc.edu`, `mohiths@usc.edu`. Update the list in `FINAL/chess_ai/game_engine/cnn.py` if you need changes.

## Run training

From `FINAL/chess_ai`:
```bash
source ../.venv/bin/activate   # adjust path to your venv
python game_engine/cnn.py      # or bash scripts/run_overnight.sh
```
