# AI Overview Satisfaction Prediction using LLMs by Understanding User Behavioral Signals

Data and code accompanying the paper **AI Overview Satisfaction Prediction using LLMs by Understanding User Behavioral Signals**.

## Overview

This work studies user satisfaction with AI Overviews presented alongside traditional ranked search results. Our method, **AI Overview Satisfaction Predictor**, uses an LLM to reason over AI Overview content, user behavior, and search context, and incorporates reinforcement learning with reward shaping based on the necessary signal set.

The repository provides the user-study dataset, baseline implementations and code references for the overall experiments, reinforcement learning code for our method, and reasoning-paradigm ablation records.

## Data

The **AI Overview User Study Dataset** was collected through a controlled user study on an experimental search platform that displayed an AI Overview together with traditional ranked results. We recruited **31 participants** to perform search tasks drawn from a set of **10 tasks** of varying difficulty, most selected from the information needs in the TREC 2002–2004 collection. Participants searched as they normally would with an AI-Overview-enabled search engine. A browser plugin recorded their interactions, and participants rated their satisfaction with the AI Overview on a **five-point scale after each query**.

The released dataset contains **243 task sessions and 554 query records**, including queries, AI Overviews and their references, traditional search results, behavioral information, and explicit satisfaction ratings.

For the study design, dataset statistics, and data organization, see the [dataset documentation](data/data_readme.md).

## Repository Structure

```text
.
├── README.md
├── data/
│   ├── data_readme.md                           # Dataset documentation
│   └── new ai overview user study dataset.json  # User-study dataset
└── code/
    ├── baselines/                              # Overall experiment baselines
    ├── our method/                             # Proposed method
    └── ablation study/                         # Reasoning-paradigm ablation records
```

## Overall Experiments

We compare our method with three categories of baselines. For `category_1_baselines` and `category_2_baselines`, we use existing open-source code and packages, with the implementation sources listed below.

### category_1_baselines: GenIR Quality Assessment

These methods assess AI Overview quality. We use their official code or package implementations. Following their order in Table 2 of the paper, the methods and corresponding sources are:

| Method | Measures reported in Table 2 | Official code / package |
| --- | --- | --- |
| ARES | context_relevance, faithfulness, answer_relevance | [ARES](https://github.com/stanford-futuredata/ARES) |
| TruLens | context_relevance, groundedness, answer_relevance | [TruLens](https://github.com/truera/trulens) |
| RAGAS | faithfulness, answer_relevancy, context_relevancy | [RAGAS](https://github.com/vibrantlabsai/ragas) |
| OpenEval | faithfulness | [Towards Lighter and Robust Evaluation](https://github.com/Razvanip13/Towards_Lighter_And_Robust_Evaluation) |

### category_2_baselines: GenIR Satisfaction Evaluation

For the following three methods, we reuse Shi et al.'s implementations in [llm_judge_baselines.py](https://github.com/Academic-Hammer/G-CoS/blob/main/src/llm_baselines/llm_judge_baselines.py):

- **G-Eval** (G-Eval-Sat in the paper): chain-of-thought satisfaction scoring.
- **URS (User Response Satisfaction):** task-type-aware multi-dimensional scoring.
- **GAH:** five-dimension scoring covering coherence, coverage, consistency, correctness, and clarity.

### category_3_baselines: Interaction-Based Satisfaction Prediction

Our implementations are available in [code/baselines/category_3_baselines/](code/baselines/category_3_baselines/), organized into two groups:

**1. Heuristic methods**

- **Click-Reformulation Heuristic:** Rule-based satisfaction scoring from post-AIO clicks and subsequent query reformulation.
- **Dwell-time Heuristic:** Rule-based scoring from AI Overview dwell time, adjusted for clicks and query reformulation.

**2. Machine learning methods**

We include behavior-based satisfaction prediction methods commonly used in traditional IR evaluation: **Ridge, Random Forest, GBDT, MLP, LSTM, and Transformer**. These methods use user interaction features or sequences, with AI Overview usefulness as an auxiliary signal.

We also include **G-CoS**, adapted from its [official implementation](https://github.com/Academic-Hammer/G-CoS/blob/main/src/models/gcos_model.py).

### Our Method

The [our method/](code/our%20method/) directory contains the `verl`-based reinforcement learning code, including custom reward scoring and reference signal label generation.

## Ablation Study

The [ablation study/](code/ablation%20study/) directory contains JSONL records for the full reasoning paradigm and four variants that remove behavioral signals, content signals, contextual signals, or structured signal-wise reasoning. Each file includes prompts, model outputs, and satisfaction labels.
