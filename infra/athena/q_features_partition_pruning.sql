SELECT block_id, COUNT(*) AS n_rows, SUM(label) AS n_positives,
               AVG(CAST(label AS DOUBLE)) AS prevalence
        FROM amazonia_deforestation.features_by_block
        WHERE block_id IN (100, 200, 300, 400)
        GROUP BY block_id
        ORDER BY block_id
