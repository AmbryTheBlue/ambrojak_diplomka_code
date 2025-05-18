# Anomaly Detection in Metrans Transport Orders
### Misclassification Detection, Confidence-Based Anomaly Detection and Frequency-Based Outlier Detection with Back-off Smoothing in Textual Tabular Data
## Bc. Jakub Ambroz

This work is part of my diploma thesis that can be seen in full [https://github.com/AmbryTheBlue/ambrojak_diplomka](https://github.com/AmbryTheBlue/ambrojak_diplomka)

### Orientation
If you are reading this in private Metrans' private gitlab the files here are here in full and should be runnable in JupyterHub. If you are located in (public GitHub repository)[https://github.com/AmbryTheBlue/ambrojak_diplomka_code] please not that there have been several redactions to obscure the access to the database, and specific table and column names.

* src/preprocessors folder has custom TextPreprocessor and TargetColumnPreprocessor
* src/oodd_detectors includes several iterations of attempts at FBOD classifier, the most recommended (and most recent) are in counter_OODD.py
* Out of the jupyter notebooks in the root directory the recommended example is chassis.ipynb

### Future Work
* Refactoing: Clean up old notebooks
* Refactoring: Organize function better into files
* Refactoring: rename OODD (Out Of Distribtution Detectors) to FBOD (Frequency Based Otlier Detection), and similarly FallBack -> BackOff Smoothing
* Refactoring: rename class_predictors -> MD (Misclassification Detection)
* Refactoring: rename probs_predictors -> CBAD (Confidence Based Anomaly Detection)
* Unify the interface to each of these approaches
* Use this unified interface to improve integration with mlflow
* Simplify adding new anomaly type
* Wrap entire workflow into function with input anomaly type, that tries several approaches and models, picks the best one and uploads it to mlflow, and the location of the model is the ouput
* The previous point fully prepares it to periodic manual rescpraing