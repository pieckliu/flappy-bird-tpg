param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$weightedPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pytpgPython = Join-Path $projectRoot "pytpg_flappy\.venv\Scripts\python.exe"
$safeEntry = Join-Path $projectRoot "run_comparison_safe.py"
$plotScript = Join-Path $projectRoot "plot_mean_fitness.py"
$output = Join-Path $projectRoot "experiments\tpg_comparison"
$retryDirectory = Join-Path $output "standard_tpg\seed_0000200042"

$runs = 10
$generations = 200
$population = 80
$episodes = 3
$validationEpisodes = 5
$testEpisodes = 100
$maxSteps = 6000
$baseSeed = 42
$seedStride = 100000
$testSeed = 100000000

function Get-CsvRowCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return 0
    }
    return @(Import-Csv -LiteralPath $Path).Count
}

function Invoke-CheckedPython {
    param(
        [string]$Python,
        [string[]]$Arguments
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python process exited with code $LASTEXITCODE"
    }
}

Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Path $output -Force | Out-Null

$retryHistory = Join-Path $retryDirectory "history.csv"
$retryTests = Join-Path $retryDirectory "test_episodes.csv"
$retryCheckpoint = Join-Path $retryDirectory "best_agent.pkl"
$retryComplete = (
    (Test-Path -LiteralPath $retryCheckpoint -PathType Leaf) -and
    (Get-CsvRowCount $retryHistory) -ge $generations -and
    (Get-CsvRowCount $retryTests) -ge $testEpisodes
)

if (-not $retryComplete) {
    Write-Output "Rerunning interrupted standard_tpg seed=200042 from generation 1"
    New-Item -ItemType Directory -Path $retryDirectory -Force | Out-Null
    $retryArguments = @(
        $safeEntry,
        "--worker", "standard_tpg",
        "--run-seed", "200042",
        "--run-dir", $retryDirectory,
        "--generations", "$generations",
        "--population", "$population",
        "--episodes", "$episodes",
        "--validation-episodes", "$validationEpisodes",
        "--test-episodes", "$testEpisodes",
        "--max-steps", "$maxSteps",
        "--test-seed", "$testSeed"
    )
    & $pytpgPython @retryArguments 2>&1 |
        Tee-Object -FilePath (Join-Path $retryDirectory "run.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Retry worker exited with code $LASTEXITCODE"
    }
}
else {
    Write-Output "Skipping complete run: standard_tpg, seed=200042"
}

Write-Output "Starting/resuming the full 10-seed comparison"
$coordinatorArguments = @(
    $safeEntry,
    "--runs", "$runs",
    "--generations", "$generations",
    "--population", "$population",
    "--episodes", "$episodes",
    "--validation-episodes", "$validationEpisodes",
    "--test-episodes", "$testEpisodes",
    "--max-steps", "$maxSteps",
    "--base-seed", "$baseSeed",
    "--seed-stride", "$seedStride",
    "--test-seed", "$testSeed",
    "--output", $output,
    "--weighted-python", $weightedPython,
    "--pytpg-python", $pytpgPython,
    "--plot-x", "generation",
    "--continue-on-error"
)
Invoke-CheckedPython -Python $weightedPython -Arguments $coordinatorArguments

$incomplete = @()
for ($runIndex = 0; $runIndex -lt $runs; $runIndex++) {
    $seed = $baseSeed + $runIndex * $seedStride
    $seedName = "seed_{0:D10}" -f $seed
    foreach ($algorithm in @("weighted_tpg", "standard_tpg")) {
        $runDirectory = Join-Path (Join-Path $output $algorithm) $seedName
        $historyRows = Get-CsvRowCount (Join-Path $runDirectory "history.csv")
        $testRows = Get-CsvRowCount (Join-Path $runDirectory "test_episodes.csv")
        $checkpointName = if ($algorithm -eq "weighted_tpg") {
            "best.json"
        }
        else {
            "best_agent.pkl"
        }
        $checkpointExists = Test-Path -LiteralPath (
            Join-Path $runDirectory $checkpointName
        ) -PathType Leaf
        if (
            $historyRows -lt $generations -or
            $testRows -lt $testEpisodes -or
            -not $checkpointExists
        ) {
            $incomplete += (
                "$algorithm seed=$seed generations=$historyRows " +
                "tests=$testRows checkpoint=$checkpointExists"
            )
        }
    }
}

if ($incomplete.Count -gt 0) {
    Write-Output "The following tasks still require attention:"
    $incomplete | ForEach-Object { Write-Output $_ }
    throw "$($incomplete.Count) training task(s) are incomplete"
}

$generationCsv = Join-Path $output "all_generations.csv"
$fitnessFigure = Join-Path $output "mean_fitness_by_generation.png"
Invoke-CheckedPython -Python $weightedPython -Arguments @(
    $plotScript,
    "--input", $generationCsv,
    "--output", $fitnessFigure
)

$completedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
Write-Output "Formal experiment complete at $completedAt"
Write-Output "Mean-fitness figure: $fitnessFigure"
