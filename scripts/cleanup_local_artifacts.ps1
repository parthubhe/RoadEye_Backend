param([switch]$Apply, [switch]$IncludeVerifiedModels, [switch]$RemoveEnvironment)
$ErrorActionPreference = 'Stop'
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$auditDir = Join-Path $project 'cleanup_archive'
if (-not (Test-Path -LiteralPath (Join-Path $auditDir 'local_inventory.json'))) { throw 'Run the inventory first.' }
$weightExtensions = @('.pt','.pth','.ckpt','.onnx','.safetensors','.tflite','.engine')
$protected = @{}
git -C $project ls-files | ForEach-Object { $protected[[System.IO.Path]::GetFullPath((Join-Path $project $_))] = $true }
$directories = @(
 'Datasets','runs','assets','dataset_crack/runs','dataset_crack/notebook/runs','dataset_crack/notebook/workspace/weights',
 'dataset_crack/notebook/workspace/dataset',
 'dataset_crack/final_dataset_v2','dataset_crack/final_dataset_v3','dataset_crack/final_dataset_v4',
 'dataset_crack/final_dataset_v4_clahe','dataset_crack/final_dataset_v5','dataset_crack/final_dataset_v5_clahe',
 'dataset_crack/final_dataset_v5_no_sediment','dataset_crack/final_dataset_v5_source_rebuild_backup',
 'dataset_crack/final_dataset_v6','dataset_crack/v5_error_filtered_diagnostic','dataset_crack/closeup_crack_v4',
 'dataset_crack/merge_staging','dataset_crack/remapped','dataset_crack/sewer_remapped',
 'dataset_crack/sewer_remapped_clahe','dataset_crack/sewer.v2-sw2-2.2k.yolo26',
 'dataset_crack/rfdetr_local_workspace','dataset_crack/distillation_outputs',
 'dataset_crack/hybrid_crack_masks','dataset_crack/sam31_preview','dataset_crack/presentation_assets',
 'dataset_crack/crack_example_review','dataset_crack/dry_training_example_review',
 'dataset_crack/training_examples_5_no_visible_water','dataset_crack/training_examples_5_per_class',
 'dataset_crack/training_examples_final_25','dataset_crack/training_examples_rfdetr50_20261027_084336',
 'dataset_crack/webapp/data'
)
$files = [System.Collections.Generic.Dictionary[string,System.IO.FileInfo]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($relative in $directories) {
 $target = [System.IO.Path]::GetFullPath((Join-Path $project $relative))
 if (-not $target.StartsWith($project + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Out-of-workspace target: $target" }
 if (-not (Test-Path -LiteralPath $target)) { continue }
 # File symlinks in the local HF cache are unlinked as files; never traverse directory links.
 $links = @(Get-ChildItem -LiteralPath $target -Directory -Recurse -Force -Attributes ReparsePoint)
 if ($links.Count) { throw "Refusing to traverse reparse points in $target" }
 Get-ChildItem -LiteralPath $target -File -Recurse -Force | ForEach-Object {
  if ($weightExtensions -contains $_.Extension.ToLowerInvariant()) { return }
  if ($protected.ContainsKey($_.FullName) -and $_.Extension -ne '.cache') { return }
  $files[$_.FullName] = $_
 }
}
foreach ($base in @($project, (Join-Path $project 'dataset_crack'))) {
 Get-ChildItem -LiteralPath $base -File | Where-Object { $_.Extension -in @('.zip','.rar','.7z') } | ForEach-Object { $files[$_.FullName] = $_ }
}
Get-ChildItem -LiteralPath $project -Directory -Recurse -Force | Where-Object {
 $_.Name -eq '__pycache__' -and -not $_.FullName.StartsWith((Join-Path $project 'venv')+'\') -and -not $_.FullName.Contains('\.git\')
} | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -File -Recurse | ForEach-Object { $files[$_.FullName] = $_ } }
if ($IncludeVerifiedModels) {
 $verifiedFile = Join-Path $auditDir 'verified_model_backups.json'
 if (-not (Test-Path -LiteralPath $verifiedFile)) { throw 'No completed hash-verified model backup manifest; refusing model deletion.' }
 $verified = Get-Content -LiteralPath $verifiedFile -Raw | ConvertFrom-Json
 if (-not $verified.verified_remote_revision) { throw 'Missing remote verification revision.' }
 foreach ($model in $verified.weights) {
  $target = [System.IO.Path]::GetFullPath((Join-Path $project $model.path))
  if (-not $target.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Model path outside workspace.' }
  if (Test-Path -LiteralPath $target) {
   if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $model.sha256) { throw "Model changed since backup: $target" }
   $files[$target] = Get-Item -LiteralPath $target
  }
 }
}
$total = ($files.Values | Measure-Object -Property Length -Sum).Sum
Write-Output ('Selected {0} files, {1:N2} GiB logical size. Source, reports, and tracked metadata are retained.' -f $files.Count, ($total/1GB))
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$plan = Join-Path $auditDir "deletion_plan_$timestamp.csv"
$files.Values | Select-Object FullName,Length | Export-Csv -LiteralPath $plan -NoTypeInformation
if (-not $Apply) { Write-Output "Dry run: $plan"; return }
$log = Join-Path $auditDir "deleted_$timestamp.csv"
$removed=0
$writer = [System.IO.StreamWriter]::new($log, $false, [System.Text.UTF8Encoding]::new($false))
$writer.WriteLine('Path,Bytes')
try {
 foreach ($file in $files.Values) {
  if (-not (Test-Path -LiteralPath $file.FullName)) { continue }
  Remove-Item -LiteralPath $file.FullName -Force
  $writer.WriteLine(('"{0}",{1}' -f $file.FullName.Replace('"','""'),$file.Length))
  $removed++
  if ($removed % 10000 -eq 0) { $writer.Flush(); Write-Output "Deleted $removed files" }
 }
} finally {
 $writer.Dispose()
}
if ($RemoveEnvironment) {
 if (-not $IncludeVerifiedModels) { throw 'Environment removal requires completed model verification.' }
 $environmentPath = [System.IO.Path]::GetFullPath((Join-Path $project 'venv'))
 if ($environmentPath -ne ($project+'\venv')) { throw 'Unexpected environment path.' }
 if (Test-Path -LiteralPath $environmentPath) { Remove-Item -LiteralPath $environmentPath -Recurse -Force }
 Write-Output 'Removed local venv; recreate dependencies before running the server.'
}
foreach ($relative in $directories) {
 $target = [System.IO.Path]::GetFullPath((Join-Path $project $relative))
 if (-not $target.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid prune target.' }
 if (-not (Test-Path -LiteralPath $target)) { continue }
 Get-ChildItem -LiteralPath $target -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
  if (-not ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -and @(Get-ChildItem -LiteralPath $_.FullName -Force).Count -eq 0) {
   Remove-Item -LiteralPath $_.FullName -Force
  }
 }
 if (@(Get-ChildItem -LiteralPath $target -Force).Count -eq 0) { Remove-Item -LiteralPath $target -Force }
}
Write-Output "Removed $removed files. Audit log: $log"
