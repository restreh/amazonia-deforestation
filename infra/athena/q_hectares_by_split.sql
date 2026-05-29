SELECT split_code, model,
               SUM(predicted_ha) AS predicted_total_ha,
               SUM(truth_ha) AS truth_total_ha,
               CASE WHEN SUM(truth_ha) > 0
                    THEN SUM(predicted_ha) / SUM(truth_ha)
                    ELSE NULL END AS ratio_pred_truth
        FROM amazonia_deforestation.metrics_by_block
        WHERE split_code IN ('val', 'test')
        GROUP BY split_code, model
        ORDER BY split_code, model
