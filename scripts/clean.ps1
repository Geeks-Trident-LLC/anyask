<#
.SYNOPSIS
    Remove Python caches, build artifacts, coverage files, and Ruff caches.
.DESCRIPTION
    Single-pass directory walk: once a directory matches a "delete whole
    subtree" pattern (__pycache__, .tox, build, etc.) it is removed and
    never descended into, and .git is skipped entirely. This avoids
    re-scanning the tree once per pattern and avoids walking into large
    directories (site-packages under .tox, .git objects) that are about
    to be deleted or are never relevant anyway.
#>

# Directory names whose entire subtree is deleted without being descended into
$pruneDirNames = @(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ruffcache",
    ".tox",
    ".mypy_cache",
    "htmlcov",
    "build",
    "dist"
)
$pruneDirPatterns = @("*.egg-info")

# File name patterns removed wherever they're found
$fileDeletePatterns = @("*.pyc", ".coverage", "coverage.xml", "*.whl", "*.tar.gz")

# Directories never worth descending into
$skipDirNames = @(".git")

function Test-ShouldPruneDir([string]$name) {
    if ($pruneDirNames -contains $name) { return $true }
    foreach ($pattern in $pruneDirPatterns) {
        if ($name -like $pattern) { return $true }
    }
    return $false
}

function Test-ShouldDeleteFile([string]$name) {
    foreach ($pattern in $fileDeletePatterns) {
        if ($name -like $pattern) { return $true }
    }
    return $false
}

function Remove-CachesUnder([string]$root) {
    $stack = [System.Collections.Generic.Stack[string]]::new()
    $stack.Push($root)

    while ($stack.Count -gt 0) {
        $current = $stack.Pop()
        $entries = Get-ChildItem -LiteralPath $current -Force -ErrorAction SilentlyContinue
        foreach ($entry in $entries) {
            if ($entry.PSIsContainer) {
                if ($skipDirNames -contains $entry.Name) {
                    continue
                }
                if (Test-ShouldPruneDir $entry.Name) {
                    Remove-Item -LiteralPath $entry.FullName -Recurse -Force -ErrorAction SilentlyContinue
                    continue
                }
                $stack.Push($entry.FullName)
            }
            elseif (Test-ShouldDeleteFile $entry.Name) {
                Remove-Item -LiteralPath $entry.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "Cleaning Python caches and build artifacts..."
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Remove-CachesUnder (Get-Location).Path
$stopwatch.Stop()
Write-Host ("Cleanup complete in {0:N2}s." -f $stopwatch.Elapsed.TotalSeconds)
