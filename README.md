# Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension

This repository contains the code and dataset for the **AAAI 2025** paper:

**"Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension"**

## Data

You can download the dateset from https://huggingface.co/datasets/lalalamdbf/TPReasoner-PODA. Please put it in ./train/data.

```
reclor
- reclor_train_origin.json  # original data
- reclor_train_all.json # original data and synthetic data
- reclor_comparsion.json # for TPCL
- reclor_val.json
- reclor_test.json
```

```
logiqa
- logiqa_train_origin.json # original data
- logiqa_train_all.json # original data and synthetic data
- logiqa_comparsion.json # for TPCL
- logiqa_val.json
- logiqa_test.json
```

## Installation

Ensure you have all dependencies installed by running:

```bash
pip install -r requirements.txt
```

## Premise-Oriented Data Augmentation (PODA)

Navigate to the PODA directory:

```bash
cd ./PODA
```

Replace `your_api_key` with your actual API key in all necessary files.

### Running the Pipeline

#### 1. CoT Rationale Annotation

```bash
python cot_rationale_annotation.py \
    --type reclor \
    --save_dir ./data/reclor \
    --train_file ./data/reclor_train.json \
    --incontext_file ./data/Incontext_exemplar.json
```

#### 2. Extract In-context Exemplars

```bash
python extract_incontext.py \
    --train_cot_file ./data/reclor/reclor_origin_cot_example.json \
    --train_file ./data/reclor_train.json \
    --output_premises_all ./data/reclor/reclor_incontext_context.json \
    --output_premises_answer ./data/reclor/reclor_incontext_premises.json
```

#### 3. Premises Generation

```bash
python premises_generation.py \
    --type reclor \
    --train_file ./data/reclor_train.json \
    --incontext_file ./data/reclor/reclor_incontext_premises.json \
    --save_dir_path ./data/reclor
```

#### 4. Organize Premises

```bash
python organize_premises.py \
    --train_cot_file ./data/reclor/reclor_origin_cot_example.json \
    --counter_premises_file ./data/reclor/reclor_counter_premises.json \
    --output_file ./data/reclor/reclor_premises_for_context.json
```

#### 5. Context Generation

```bash
python context_generation.py \
    --type reclor \
    --train_file ./data/reclor_train.json \
    --counter_file ./data/reclor/reclor_premises_for_context.json \
    --incontext_file ./data/reclor/reclor_incontext_context.json \
    --save_dir_path ./data/reclor
```

#### 6. Correctness Verification

```bash
python correctness_verfication.py \
    --type reclor \
    --save_dir_path ./data/reclor \
    --counter_train_file ./data/reclor/reclor_counter_context.json \
    --incontext_file ./data/Incontext_exemplar.json
```

## Training

Navigate to the training directory:

```bash
cd ./train
```

### Train Models

#### Train with Original Data

```bash
sh reclor_train_baseline.sh
sh logiqa_train_baseline.sh
```

#### Train with Counterfactual Data

```bash
sh reclor_train_cd.sh
sh logiqa_train_cd.sh
```

#### Merge LoRA for SFT Model

```bash
sh merge_lora.sh
```

#### Train with Thought-Path Contrastive Learning (TPCL)

```bash
sh reclor_tpcl.sh
sh logiqa_tpcl.sh
```

#### Generate Predictions

```bash
sh predict.sh
```

## Evaluation

### Evaluate Model Performance

#### For `reclor_val`, `logiqa_val`, and `logiqa_test`

```bash
python ./src/calculate_acc.py --input_file predicted_file
```

#### For `reclor_test`

```bash
python ./src/extract_answer_for_test.py --input_file predicted_file --out_file predicted.npy
```

### Evaluate Data Quality

Navigate to the evaluation directory:

```bash
cd ./evaluation
```

#### Context Evaluation

##### Coherence
```bash
python evaluate_coherence.py \
    --input_file ./data/context/samples_200_counter_poda.json \
    --output_file ./data/context/output/counter_poda_coherence.json

python evaluate_coherence.py \
    --input_file ./data/context/samples_200_counter_lr.json \
    --output_file ./data/context/output/counter_lr_coherence.json
```

##### Clarity
```bash
python evaluate_clarity.py \
    --input_file ./data/context/samples_200_counter_poda.json \
    --output_file ./data/context/output/counter_poda_clarity.json

python evaluate_clarity.py \
    --input_file ./data/context/samples_200_counter_lr.json \
    --output_file ./data/context/output/counter_lr_clarity.json
```

##### Relevance
```bash
python evaluate_relevance.py \
    --input_file ./data/context/samples_200_counter_poda.json \
    --qo_file ./data/context/samples_200_qo.json \
    --output_file ./data/context/output/counter_poda_relevance.json

python evaluate_relevance.py \
    --input_file ./data/context/samples_200_counter_lr.json \
    --qo_file ./data/context/samples_200_qo.json \
    --output_file ./data/context/output/counter_lr_relevance.json
```

##### Diversity
```bash
python evaluate_diversity.py \
    --counter_file ./data/context/samples_200_counter_poda.json \
    --origin_file ./data/context/samples_200_origin.json \
    --output_file ./data/context/output/counter_poda_diversity.json

python evaluate_diversity.py \
    --counter_file ./data/context/samples_200_lr_poda.json \
    --origin_file ./data/context/samples_200_origin.json \
    --output_file ./data/context/output/counter_lr_diversity.json
```

#### Chain of Thought (CoT) Evaluation

##### Coherence
```bash
python evaluate_cot_coherence.py \
    --input_file ./data/cot/samples_200_poda.json \
    --output_file ./data/cot/output/counter_poda_coherence.json

python evaluate_cot_coherence.py \
    --input_file ./data/cot/samples_200_logicot.json \
    --output_file ./data/cot/output/counter_logicot_coherence.json
```

##### Completeness, Relevance, and Faithfulness
```bash
python evaluate_cot_completeness.py --input_file ./data/cot/samples_200_poda.json --output_file ./data/cot/output/counter_poda_completeness.json
python evaluate_cot_relevance.py --input_file ./data/cot/samples_200_poda.json --output_file ./data/cot/output/counter_poda_relevance.json
python evaluate_cot_faithfulness.py --input_file ./data/cot/samples_200_poda.json --output_file ./data/cot/output/counter_poda_faithfulness.json
```

### Calculate Final Score

```bash
python calculate_score.py --file_path waiting_to_calculate_file
```

## Bibliography

If you find this repo useful, please cite our paper.

```
@article{wang2024thought,
  title={Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension},
  author={Wang, Chenxu and Jian, Ping and Yang, Zhen},
  journal={arXiv preprint arXiv:2409.14495},
  year={2024}
}
```
