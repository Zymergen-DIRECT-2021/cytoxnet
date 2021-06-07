# Dataset reports
A few datasets were identified as potentially useful, formated and included in the package. This directory contains a report on each of those datasets, inclusing information such as distribution, target analysis, and overall comments. All data contained in the package is in log toxicity units.

## Contents
- `chembl_ecoli.ipynb`: analysis if the ChemBL E. coli query
- `lunghini_ecotoxological.ipynb`: analysis of Lunghini's fish, daphnia, algea toxicity data
- `zhu_rat.ipynb`: analysis of Zhu's rat LD50 data
- `dataset_overlap.ipynb`: analysis of overlap of the datasets in feature space

## Datasets
Three datasets containing 5 species were identified for the investigation:

ChemBL E. coli:
> https://www.ebi.ac.uk/chembl/. Query details - XXXXXXXXX

Contains 5217 MIC datum from a number of sources for E. coli. Data is logged and is stored units of MIC \[log(ug/mL)\].

Zhu Rat LD50:
> Zhu, Hao, et al. “Quantitative structure− activity relationship modeling of rat acute toxicity by oral exposure.” Chemical research in toxicology 22.12 (2009): 1913-1921.

Contains 7384 LD50 datum for rats. Data is stored in units of LD50 \[log(mol/kg)\]

Lunghini fish, daphnia, algea:
> F. Lunghini, G. Marcou, P. Azam, M. H. Enrici, E. Van Miert, and A. Varnek, “Consensus QSAR models estimating acute toxicity to aquatic organisms from different trophic levels: algae, Daphnia and fish,” SAR QSAR Environ. Res., vol. 31, no. 9, pp. 655–675, Sep. 2020, doi: 10.1080/1062936X.2020.1797872.

Contains 2199 LC50 datum for fish, 2107 EC50 datum for daphnia, and 1440 EC50 datum for algea. All units are stored in units of \[log(mg/L)\]

Preliminary cleaning was conducted on the datasets, such as canonicalization, removal of ivalid smiles, and ensuring logarithmic data storage before depositing into the package data.