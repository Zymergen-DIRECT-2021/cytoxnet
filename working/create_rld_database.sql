.headers on
.mode csv
.output RLD_database.csv

SELECT RLD.id AS id, COMPOUNDS.id AS comp_id, RLD.compound_name AS compound_name, RLD.species AS species, RLD.ld50 AS ld50, RLD.units AS units, RLD.source AS source
FROM RLD
INNER JOIN COMPOUNDS ON COMPOUNDs.smiles = RLD.smiles
WHERE RLD.id IS NOT NULL
;