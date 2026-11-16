# Unpack wheels into a venv with .NET (workaround: Python/pip extraction hangs on this machine, see README).
param(
    [Parameter(Mandatory)][string]$Venv,
    [Parameter(Mandatory)][string]$Wheels
)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$site = Join-Path $Venv "Lib\site-packages"
$scripts = Join-Path $Venv "Scripts"
$sw = [Diagnostics.Stopwatch]::StartNew()
$all = Get-ChildItem $Wheels -Filter *.whl | Sort-Object { $_.Name -like "torch-*" }, Name   # torch last
foreach ($w in $all) {
    $t0 = $sw.Elapsed.TotalSeconds; $n = 0
    $zip = [IO.Compression.ZipFile]::OpenRead($w.FullName)
    try {
        foreach ($e in $zip.Entries) {
            if ($e.FullName.EndsWith("/")) { continue }
            $rel = $e.FullName
            if ($rel -match '^[^/]+\.data/(purelib|platlib)/(.*)$') { $dest = Join-Path $site $Matches[2] }
            elseif ($rel -match '^[^/]+\.data/scripts/(.*)$') { $dest = Join-Path $scripts $Matches[1] }
            elseif ($rel -match '^[^/]+\.data/data/(.*)$') { $dest = Join-Path $Venv $Matches[1] }
            elseif ($rel -match '^[^/]+\.data/headers/') { continue }
            else { $dest = Join-Path $site $rel }
            $dir = Split-Path $dest -Parent
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $dest, $true)
            $n++
        }
    } finally { $zip.Dispose() }
    "{0,7:N1}s  {1,6} files  {2}" -f ($sw.Elapsed.TotalSeconds - $t0), $n, $w.Name
}
"DONE in {0:N1}s" -f $sw.Elapsed.TotalSeconds
