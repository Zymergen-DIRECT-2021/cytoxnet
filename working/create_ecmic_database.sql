.headers on
.mode csv
.output ECMIC_database.csv

SELECT ECMIC.id AS id, COMPOUNDS.id AS comp_id, ECMIC.molecule_chembl_id AS molecule_chembl_id, ECMIC.species AS species, ECMIC.mic AS mic, ECMIC.units AS units, ECMIC.censoring AS censoring, ECMIC.molecular_weight AS molecular_weight, ECMIC.assay_parameters AS assay_parameters, ECMIC.source AS source, ECMIC.source_journal AS source_journal
FROM ECMIC
INNER JOIN COMPOUNDS ON COMPOUNDs.smiles = ECMIC.smiles
WHERE ECMIC.id IS NOT NULL
;