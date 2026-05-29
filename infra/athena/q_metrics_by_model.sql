SELECT model,
               AVG(f1) AS mean_f1,
               AVG(iou) AS mean_iou,
               AVG(precision) AS mean_precision,
               AVG(recall) AS mean_recall,
               SUM(tp) AS total_tp, SUM(fp) AS total_fp,
               SUM(fn) AS total_fn, SUM(tn) AS total_tn
        FROM amazonia_deforestation.metrics_by_block
        WHERE split_code = 'test'
        GROUP BY model
        ORDER BY mean_f1 DESC
