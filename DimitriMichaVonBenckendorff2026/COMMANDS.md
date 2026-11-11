# Commands to run the Python scripts

## Literature extractions

after installing argpase, run
```[python3]
python .\analyze-extractions-first-pass.py --path .\Data\extraction-first-pass.xlsx --outdir .\analysis_out
```
to cluster the first pass coding results, and run
```[python3]
python .\analyze-extractions-second-pass.py .\Data\extraction-second-pass.xlsx -o .\analysis_out\appendix_tables.tex
```
to check coverage on the second pass coding results.

## Expert evaluations

after installing pdfplumber, run
```[python3]
python .\analyze-evaluations.py .\Data\evaluations.pdf analysis_out
```
in the terminal to quantitatively analyze the expert interview results.