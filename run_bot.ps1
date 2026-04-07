param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$python = "C:\ProgramData\anaconda3\python.exe"

if (-not (Test-Path $python)) {
    throw "Expected Conda Python was not found at $python"
}

& $python -m trading_bot.main @ArgsList
