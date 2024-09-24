# Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension

## Data

The data for reclor and logiqa 2.0 is in ./train/data. You can unzip them in this directory.

```
reclor.zip
- reclor_train_origin.json  # original data
- reclor_train_all.json # original data and synthetic data
- reclor_comparsion.json # for TPCL
- reclor_val.json
- reclor_test.json
```

```
logiqa.zip
- logiqa_train_origin.json # original data
- logiqa_train_all.json # original data and synthetic data
- logiqa_comparsion.json # for TPCL
- logiqa_val.json
- logiqa_test.json
```

The code for data processing, generation, and training is under preparation.

## Bibliography

If you find this repo useful, please cite our paper.

```
@article{wang2024thought,
  title={Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension},
  author={Wang, Chenxu and Jian, Ping and Huang, Mu},
  journal={arXiv preprint arXiv:2409.14495},
  year={2024}
}
```

