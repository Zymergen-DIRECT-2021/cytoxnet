.headers on
.mode csv
.output fillipo_database.csv

SELECT FILLIPO.id AS id, COMPOUNDS.id AS comp_id, FILLIPO.species AS species, FILLIPO.daphnia_ec50 AS daphnia_ec50, FILLIPO.fish_lc50 AS fish_lc50, FILLIPO.algea_ec50 AS algea_ec50, FILLIPO.units AS units, FILLIPO.source AS source
FROM FILLIPO
INNER JOIN COMPOUNDS ON COMPOUNDs.smiles = FILLIPO.smiles
WHERE FILLIPO.id IS NOT NULL
;