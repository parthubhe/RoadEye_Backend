param([switch]$Publish)

$ErrorActionPreference = 'Stop'
$repo = 'AkumaDachi/roadeeye-sewer-defects-v5'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$rfRun = 'dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7'

$files = @(
    @('dataset_crack/models/README.md', 'models/README.md'),
    @('dataset_crack/models/classwise_validation_summary.json', 'models/classwise_validation_summary.json'),
    @("$rfRun/checkpoint_best_total.pth", 'models/rfdetr_nano_50e/checkpoint_best_total.pth'),
    @("$rfRun/best_validation_metrics.json", 'models/rfdetr_nano_50e/best_validation_metrics.json'),
    @("$rfRun/full_v5_test_metrics.json", 'models/rfdetr_nano_50e/full_v5_test_metrics.json'),
    @("$rfRun/full_v5_test_evaluation.json", 'models/rfdetr_nano_50e/full_v5_test_evaluation.json'),
    @("$rfRun/dataset_provenance.json", 'models/rfdetr_nano_50e/dataset_provenance.json'),
    @("$rfRun/roadeeye_experiment.json", 'models/rfdetr_nano_50e/roadeeye_experiment.json'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/weights/best.pt', 'models/yolo26s_v5_baseline/best.pt'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_cctv/args.yaml', 'models/yolo26s_v5_baseline/args.yaml'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/weights/best.pt', 'models/yolo26s_v5_clahe_simam/best.pt'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_simam/args.yaml', 'models/yolo26s_v5_clahe_simam/args.yaml'),
    @('dataset_crack/generated_attention_models/yolo26s_simam_v5.yaml', 'models/yolo26s_v5_clahe_simam/model.yaml'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/weights/best.pt', 'models/yolo26s_v5_clahe_cbam/best.pt'),
    @('dataset_crack/runs/pipe_proto/yolo26s_v5_clahe_cbam/args.yaml', 'models/yolo26s_v5_clahe_cbam/args.yaml'),
    @('dataset_crack/generated_attention_models/yolo26s_cbam_v5.yaml', 'models/yolo26s_v5_clahe_cbam/model.yaml'),
    @('dataset_crack/attention_modules.py', 'models/support/attention_modules.py')
)

foreach ($pair in $files) {
    $local = Join-Path $projectRoot $pair[0]
    if (-not (Test-Path -LiteralPath $local -PathType Leaf)) {
        throw "Required release file is missing: $local"
    }
    Write-Output "$($pair[0]) -> $($pair[1])"
}
$e3Folder = Join-Path $projectRoot 'dataset_crack/models/p2_weighted_fusion'
foreach ($relative in @('best.pt', 'test_summary.json', 'validation_summary.json', 'support/modules.py', 'support/architectures.py', 'support/experiment_trainer.py', 'support/yolo26_base.yaml', 'support/yolo26s_E3.yaml', 'support/requirements.txt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $e3Folder $relative) -PathType Leaf)) {
        throw "Required E3 release file is missing: $relative"
    }
}
Write-Output 'dataset_crack/models/p2_weighted_fusion/ -> models/p2_weighted_fusion/'

if (-not $Publish) {
    Write-Output 'Preview only. Re-run with -Publish after reviewing the destinations.'
    return
}
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw 'The hf CLI is not installed or not on PATH. Install huggingface_hub[cli] and authenticate with hf auth login.'
}
foreach ($pair in $files) {
    $local = Join-Path $projectRoot $pair[0]
    & hf upload $repo $local $pair[1] --repo-type dataset
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $($pair[0])"
    }
}
& hf upload $repo $e3Folder 'models/p2_weighted_fusion' --repo-type dataset
if ($LASTEXITCODE -ne 0) {
    throw 'E3 folder upload failed'
}
Write-Output "Published selected model files to https://huggingface.co/datasets/$repo/tree/main/models"
