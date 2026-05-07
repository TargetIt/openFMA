# Run OpenLane flow using PowerShell (no MSYS2 path conversion)
$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\work\qpwork\github\TargetIt\openFMA"
$HomeVolare = "$env:USERPROFILE\.volare"

# Ensure PDK directory exists
New-Item -ItemType Directory -Force -Path $HomeVolare | Out-Null

# Copy latest RTL
Copy-Item "$ProjectRoot\rtl\stage4\fma_top_stage4.v" "$ProjectRoot\openlane\fma_top\src\fma_top_stage4.v" -Force

Write-Host "=== Running OpenLane Flow ==="

docker run --rm `
    -v "${ProjectRoot}:/openfma" `
    -v "${HomeVolare}:/root/.volare" `
    -e PDK_ROOT=/root/.volare/volare/sky130/versions/c6d73a35f524070e85faff4a6a9eef49553ebc2b `
    efabless/openlane:latest `
    bash -c "cd /openfma/openlane/fma_top && flow.tcl 2>&1"

Write-Host "=== Flow Complete ==="
