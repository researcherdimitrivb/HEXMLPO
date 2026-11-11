# HEXMLPO = Human-centered Explainability extension for the eXplainable Machine Learning Pipeline Ontology

This repo contains relevant files from the Human-centered Explainability extension for the eXplainable Machine Learning Pipeline Ontology (HEXMLPO) by Dimitri Micha von Benckendorff (2026).

## Citation

HEXMLPO has not been published, but is an extension of the contributions proposed in this paper:

Xhani, D., Rebelo Moreira, J. L., Van Sinderen, M., & Ferreira Pires, L. (2024). XMLPO: An Ontology for Explainable Machine Learning Pipeline. In C. Trojahn, D. Porello, & P. P. F. Barcelos (Eds.), <em>Frontiers in Artificial Intelligence and Applications</em>. IOS Press. https://doi.org/10.3233/FAIA241306

## Contents

The DimitriMichaVonBenckendorff2026 folder contains the scripts and data used to develop HEXMLPO. Most importantly, the folder contains turtle files of Dimitri's extended ontology (`hexmlpo.ttl`), the updated XMLPO (`xmlpo-updated.ttl`), operational ontology for the healthcare use case (`operationalontology.ttl`), instances reflecting the findings from the problem investigation in this study (`hexmlpo-instances.ttl`), the extended ontology axioms (`hexmlpo-axioms.ttl`), image data input and processing (`img.ttl`), and deep learning models and architectures (`xdlpo.ttl`). The problem investigation clustering and taxonomy coverage scripts are stored as .py files. The ontologies are loaded and XAI experiment results were saved as seven .json files (one for provenance and six for local explanations) before constructing a knowledge graph, that was queried to answer the competency questions in the explaining-can-screening.ipynb file.

Prerequisites of this work are Donika Xhani's XMLPO paper and Sven van der Peet's CAN screening automation script:

The DonikaXhani2023 folder contains Donika's conceptual model in a .vpp file and the ontology in a .ttl file.

The SvenVanDerPeet2026 folder contains Sven's report in a .pdf file, the script in a .py file, and the input dataset in a .csv file.

## Contacts
This space is administered by:

Dimitri Micha von Benckendorff d.m.vonbenckendorff@student.utwente.nl (GitHub: researcherdimitrivb)

Donika Xhani d.xhani@utwente.nl (GitHub: DonikaXhani)

João Luiz Rebelo Moreira j.luizrebelomoreira@utwente.nl (GitHub: jonimoreira)
