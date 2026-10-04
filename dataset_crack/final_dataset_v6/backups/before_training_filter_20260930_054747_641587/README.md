# Error-filtered diagnostic dataset

This YOLO dataset view excludes the union of images in the FP/FN CSV files. It references the original images and labels through absolute-path manifests; no originals were deleted.

Selection uses predictions from the parallel model. Scores on the retained test subset are selection-biased and must not replace full-v5 test metrics or be presented as improved generalization. FP/FN status alone is not evidence that an image or annotation is defective.

Do not move excluded test images into training. Train and validation lists are unchanged when the input CSVs contain only test images; retraining is unnecessary to measure this filtering effect.

See filter_report.json for counts and excluded_images.csv for provenance. These manifests are local references, not a standalone dataset upload.
