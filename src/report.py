import numpy as np

from symai.collect.stats import Aggregator


agg = Aggregator.load('result/aggregation.json')
agg.finalize()


# apply mean before returning values
agg.map = lambda x: np.mean(x, axis=0)


class Report:
    ics     = agg.eval_in_context_associations.test_comparison
    score   = (ics.score - ics.rand_score) / (ics.base_score - ics.rand_score)


print(Report.score)
