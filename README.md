# PsychEval: A Multi-Session and Multi-Therapy Benchmark for High-Realism AI Psychological Counselor

This repository contains the dataset and evaluation framework for PsychEval, a pioneering benchmark designed to align AI capabilities with professional psychological assessment and longitudinal treatment demands.

## Core Innovations

PsychEval shifts the focus from large-scale but shallow interactions to high-fidelity, longitudinal, and skill-aware assessment.

1. Breaking the "Single-Session" Barrier: Unlike existing datasets limited to fragmented chat logs, PsychEval simulates the full trajectory of counseling, spanning 6 to 10 sessions per client across three clinical stages: Case Conceptualization, Core Intervention, and Consolidation.

2. Theoretical Versatility (Multi-Therapy): It is the first benchmark to support 5 major therapeutic schools (CBT, Psychodynamic, Behavioral, Humanistic, Postmodernist) and an Integrative approach within a unified clinical framework.

3. Hierarchical Skill Taxonomy: We provide a massive, interpretable supervision signal with 677 Meta-skills and 4,577 Atomic skills, enabling AI agents to perform coarse-to-fine clinical reasoning and strategic goal-tracking.

4. High-Realism Data Source: Grounded in 369 authentic clinical case reports from authoritative psychology journals rather than synthetic or unstructured chat logs, ensuring peak ecological validity and clinical fidelity.

## Statistical Highlights
<img width="1858" height="538" alt="image" src="https://github.com/user-attachments/assets/9230b796-8740-409f-be66-9451a3091c85" />


<img width="1394" height="496" alt="image" src="https://github.com/user-attachments/assets/d55f0a10-9f8c-444d-ac60-bf054e557cbe" />

<img width="1334" height="458" alt="image" src="https://github.com/user-attachments/assets/86ecdb17-0b95-4418-8581-5cef89f51147" />



## Evaluation Framework

We establish a holistic assessment system utilizing 18 therapy-specific and shared metrics (e.g., WAI for alliance, CTRS for CBT competency, and SCL-90 for symptom reduction). Our results show that PsychEval achieves unprecedented clinical fidelity, nearly doubling the scores of prior models in technical adherence (e.g., CTRS: 9.19).

##  Running the Evaluation
1. Main Evaluation Script

To execute the multi-dimensional evaluation, use the following command:
```
python3 -m eval.manager.evaluation_mutil
```

2. Configuring Metrics

You can easily customize the evaluation metrics by modifying the method_cls list in the configuration file. Simply update the list with the desired metric classes to toggle specific evaluations.

3. Baseline Reproduction & Data Conversion
To reproduce results from other papers (e.g., Simpsydial), you must first convert the data format to ensure compatibility.

   Step 1: Format Conversion Run the dedicated conversion script:
      ```
   python3 manager/Simpsydial/convert_simpsydial.py
      ```
   Step 2: Run Evaluation After conversion, proceed with the main evaluation script mentioned in step 1.


  
