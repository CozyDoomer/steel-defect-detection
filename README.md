# Severstal: Steel Defect Detection

[Kaggle competition link]([https://www.kaggle.com/c/champs-scalar-coupling](https://www.kaggle.com/competitions/severstal-steel-defect-detection)

In this competition, you’ll help engineers improve the algorithm by localizing and classifying surface defects on a steel sheet.

| Dataformat   |      Metric      |  Prediction |
|----------|:-------------:|------:|
| unstructured data: image | dice coefficient | segmentation |

Addapted [this public notebook](https://www.kaggle.com/code/lightforever/severstal-mlcomp-catalyst-train-0-90672-offline/notebook) and further improved it by
* add classification model for initial reduction of test set (also a filter for possible false positives)
* training on pseudolabels
* training multiple architectures, finetuning and ensembling them
* TTA (test time augmentation)

Tried [mlcomp](https://github.com/lightforever/mlcomp) and [catalyst](https://catalyst-team.github.io/catalyst/v20.03.3/index.html) for this competition.

## Placement
__top 4%__

| leaderboard   | score | placement |
|----------|:-------------:|---------:|
| public | 0.91361 | __117/2427__ |
| private | 0.90146 | __85/2427__ |
