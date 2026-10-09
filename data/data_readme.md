# AI Overview User Study Dataset

## Overview

This dataset was collected through a controlled user study designed to investigate how users interact with and evaluate search interfaces that integrate an AI Overview with traditional ranked search results.

The study was conducted using an experimental search platform that presented an AI Overview together with traditional search results. Participants were asked to interact with the system as they normally would with an AI-Overview-enabled search engine.

The final dataset contains interaction records, AI Overview content and references, traditional search results, explicit user satisfaction labels, result-level usefulness judgments, and query- and task-level contextual annotations.

## Dataset at a Glance

| Statistic                                |             Value |
| ---------------------------------------- | ----------------: |
| Participants                             |                31 |
| Task sessions                            |               243 |
| Query records                            |               554 |
| Search tasks                             |                10 |
| Avg. queries per task session            |              2.28 |
| Traditional result records               |             4,982 |
| AI Overview reference records            |             1,662 |
| Total displayed result/reference records |             6,644 |
| Queries with at least one click          | 237 / 554 (42.8%) |
| Task sessions with at least one click    | 164 / 243 (67.5%) |
| Organic-result clicks                    |               258 |
| AI Overview reference clicks             |               135 |
| Total click events                       |               393 |

The dataset is organized hierarchically as:

```text
Participant
└── Task Session
    ├── Task-level metadata and subjective assessments
    └── Query
        ├── Query text
        ├── AI Overview
        ├── AI Overview references
        ├── Traditional ranked results
        ├── Interaction signals
        ├── Satisfaction labels
        └── Query-level annotations
```

## Study Design

### Search Tasks

The study used a set of 10 search tasks spanning different levels of cognitive complexity. Following the task taxonomy used in prior information retrieval studies, the task set contains:

- 3 **Remember** tasks
- 3 **Understand** tasks
- 4 **Analyze** tasks

The tasks also cover both computer-science and non-computer-science domains.

Most tasks were selected or adapted from the information needs in the TREC 2002--2004 collection, while additional tasks were constructed to ensure coverage of computer-science topics across cognitive-complexity categories.

### Task-Level Dataset Statistics

| Task ID | Representative Topic                                         | Task Sessions | Queries | Queries / Session |
| ------: | ------------------------------------------------------------ | ------------: | ------: | ----------------: |
|       1 | Sherlock Holmes' most famous adversary                       |            25 |      30 |              1.20 |
|       2 | Which part of the cinnamon tree produces cinnamon            |            23 |      28 |              1.22 |
|       3 | DNS and domain-name resolution                               |            26 |      36 |              1.38 |
|       4 | Uses and training of working dogs in law enforcement         |            23 |      77 |              3.35 |
|       5 | Recycling and reuse of waste tires                           |            26 |      60 |              2.31 |
|       6 | Representative research directions in artificial intelligence |            24 |      67 |              2.79 |
|       7 | Effects of ultraviolet radiation on the eyes and protection methods |            24 |      76 |              3.17 |
|       8 | Causes and prevention of railway accidents                   |            23 |      60 |              2.61 |
|       9 | Comparison of real-world recycling projects                  |            26 |      72 |              2.77 |
|      10 | Keyword retrieval vs. semantic retrieval                     |            23 |      48 |              2.09 |

The number of queries varies substantially across tasks. Simpler factual tasks generally required fewer queries, whereas more exploratory and analytical tasks tended to involve longer query sequences.

## Data Collection

### Behavioral Data Collection

A browser plugin recorded users' interaction behavior during search.

The experimental logging system recorded interaction information including:

- clicks;
- mouse interactions;
- keyboard input;
- interaction positions;
- interaction timestamps.

The released dataset contains processed behavioral information associated with displayed search results and AI Overview references, including click indicators and dwell time.

### First-Party Satisfaction Collection

Participants directly provided satisfaction judgments during the study.

At the query level, the dataset contains:

- satisfaction with the overall query/search experience;
- satisfaction with the AI Overview.

A think-aloud protocol was additionally used during the experiment to collect participants' real-time reasoning and explanations surrounding their interaction decisions and satisfaction judgments.

### Task-Level Subjective Assessments

Participants also provided subjective assessments before and after search tasks.

The released data include fields such as:

## Descriptive Statistics

The study involved **31 participants**. The final dataset contained **243 task sessions** and **554 query records**. Participants completed an average of **7.84 task sessions** and issued an average of **17.87 queries** (SD = 6.57; median = 18; range = 8--42), corresponding to **2.28 queries per task session**.

When restricting the analysis to clicked items, the median dwell time was **41.15 seconds**, while clicks on AI Overview references had a median dwell time of **38.76 seconds**. AI Overview exposure time was available for all **554 queries**. Participants were exposed to an AI Overview for an average of **45.12 seconds per query** (SD = 77.25; median = 31.54 seconds).

Overall, **237 of the 554 queries (42.8%)** involved at least one link click, including clicks on either organic search results or AI Overview references, and **164 of the 243 task sessions (67.5%)** involved at least one such click.

For `query_satisfaction`, **33.0%** of ratings were 1--3, while **67.0%** were 4--5. For `aio_satisfaction`, **35.2%** of ratings were 1--3, while **64.8%** were 4--5. The two satisfaction measures were strongly positively associated (**Spearman's ρ = 0.711, p < .001**).

## Satisfaction Statistics

Two satisfaction variables are available:

1. **Query satisfaction** — satisfaction with the overall search experience for a query.
2. **AI Overview satisfaction** — satisfaction specifically with the AI-generated overview.

Both are measured on a five-point scale.

### Query Satisfaction

Valid query-satisfaction labels are available for **554 of 554 queries (100.0%)**.

| Rating | Count |
| -----: | ----: |
|      1 |    16 |
|      2 |    70 |
|      3 |    97 |
|      4 |   246 |
|      5 |   125 |

Grouping the ratings:

- Ratings **1--3**: 33.0%
- Ratings **4--5**: 67.0%

### AI Overview Satisfaction

Valid AI Overview satisfaction labels are available for **554 of 554 queries (100.0%)**.

| Rating | Count |
| -----: | ----: |
|      1 |    28 |
|      2 |    80 |
|      3 |    87 |
|      4 |   182 |
|      5 |   177 |

Grouping the ratings:

- Ratings **1--3**: 35.2%
- Ratings **4--5**: 64.8%

### Relationship Between AI Overview and Query Satisfaction

There are **554 queries** for which both satisfaction variables are available.

Across these matched observations, AI Overview satisfaction and overall query satisfaction are strongly positively associated:

> **Spearman's ρ = 0.711, p < .001**

This correlation should be interpreted as an association between the two satisfaction judgments rather than as evidence that AI Overview satisfaction causally determines overall query satisfaction.

## Task-Level Data

Each task-session object contains the sequence of queries issued while completing that task together with task-level contextual information.

A simplified structure is:

```json
{
  "user_id": "...",
  "task": {
    "<task_id>": {
      "user_id": "...",
      "content": [
        {
          "task_id": "...",
          "query_id": "...",
          "query": "...",
          "query_satisfaction": "...",
          "ai_overview": "...",
          "ai_overview_references": [...],
          "results": [...],
          "ai_overview_exposure_time": "...",
          "annotations": [...]
        }
      ],
      "input_description": "...",
      "question_answer": "...",
      "task_satisfaction": "..."
    }
  }
}
```