SELECT block_id, f1, precision, recall, n_pixels, prevalence,
               predicted_ha, truth_ha
        FROM amazonia_deforestation.metrics_by_block
        WHERE model = 'ensemble' AND split_code = 'test'
        ORDER BY f1 DESC
        LIMIT 10
