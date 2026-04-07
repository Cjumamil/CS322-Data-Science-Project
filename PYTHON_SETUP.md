# Python Setup

This project is currently expected to run with:

`C:\ProgramData\anaconda3\python.exe`

If a fresh chat or terminal session does not know which Python to use, prefer the
repo-local launcher scripts:

```powershell
.\run_backtest.ps1 --symbol ALB --start 2025-03-31 --end 2026-03-31
.\run_bot.ps1
```

VS Code is also pointed at the same interpreter through:

`.\.vscode\settings.json`

This is mainly here so the Python path is discoverable from inside the repo
instead of relying on shell activation state.
