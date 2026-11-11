# HEXMLPO = Human-Centric Explainability Extension for the Explainable Machine Learning Pipeline Ontology

This repo contains relevant files from the Human-Centric Explainability Extension for the Explainable Machine Learning Pipeline Ontology (HEXMLPO) by Dimitri Micha von Benckendorff (2026).

## Citation

HEXMLPO has not been published, but is an extension of the contributions proposed in this paper:

Xhani, D., Rebelo Moreira, J. L., Van Sinderen, M., & Ferreira Pires, L. (2024). XMLPO: An Ontology for Explainable Machine Learning Pipeline. In C. Trojahn, D. Porello, & P. P. F. Barcelos (Eds.), <em>Frontiers in Artificial Intelligence and Applications</em>. IOS Press. https://doi.org/10.3233/FAIA241306

## Contents

The DonikaXhani2023 folder contains Donika's conceptual model in a .vpp file as well as the ontology in a .ttl file.

The DimitriMichaVonBenckendorff2026 folder contains the scripts and data used to develop HEXMLPO. Most importantly, the folder contains Dimitri's extended ontology in the hexmlpo.ttl file as well as turtle files of the updated xmlpo, operational ontology for the healthcare use case, instances reflecting the findings from the problem investigation in this study, and the ontology axioms. The problem investigation clustering and taxonomy coverage scripts are stored as .py files. The ontologies are loaded and XAI experiment results were saved as seven .json files (one for provenance and six for local explanations) before constructing a knowledge graph, that was queried to answer the competency questions in the explaining-can-screening.ipynb file.

The SvenVanDerPeet2026 folder contains Sven's internship report with a proof of concept for cardiac autonomic neuropathy screening automation in a .pdf file. The machine learning pipeline developed in this work is also provided as the main_ML_V2.py file that uses data in the GE-71_Data_Summary_Table_final.csv file.

## Contacts
This space is administered by:

Dimitri Micha von Benckendorff d.m.vonbenckendorff@student.utwente.nl (GitHub: researcherdimitrivb)

Donika Xhani d.xhani@utwente.nl (GitHub: DonikaXhani)

João Luiz Rebelo Moreira j.luizrebelomoreira@utwente.nl (GitHub: jonimoreira)
