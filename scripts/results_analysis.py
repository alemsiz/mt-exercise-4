from typing import List
import re
import pandas as pd

def get_stats(log_filename: str):
    eval_lines = []
    with open(log_filename, 'r') as f:
        for line in f:
            if ("Evaluation result" in line) and ("ppl" in line):
                eval_lines.append(line)
    return eval_lines

# extract the lines containing the reported validation perplexities from each log file
baseline_eval_lines = get_stats("../logs/baseline.log")
pre_norm_eval_lines = get_stats("../logs/pre_norm_transformer/err")
post_norm_eval_lines = get_stats("../logs/post_norm_transformer/err")

def get_perplexities(eval_lines: List[str]):
    perplexities = []
    for line in eval_lines:
        ppl = re.search(r"ppl:   ?(\d+\.\d+),", line).group(1)
        perplexities.append(float(ppl))
    return perplexities

# extract the perplexity values from each extracted line
baseline_ppls = get_perplexities(baseline_eval_lines)
pre_norm_ppls = get_perplexities(pre_norm_eval_lines)
post_norm_ppls = get_perplexities(post_norm_eval_lines)

#print(len(baseline_ppls), len(pre_norm_ppls), len(post_norm_ppls)) # check same number of data points in each log file

# create a list of steps at which the validation perplexities were recorded (every 500 steps up until step 40,500)
steps = list(range(500, 40501, 500))

# create the table with the perplexities from each model
perplexities = pd.DataFrame()
perplexities['Step'] = steps
perplexities['Baseline'] = baseline_ppls
perplexities['Pre-norm'] = pre_norm_ppls
perplexities['Post-norm'] = post_norm_ppls
perplexities.set_index('Step', inplace=True)

# plot line chart
ppl_lines = perplexities.plot.line()
ppl_lines.set_ylabel("Validation perplexity")
ppl_fig = ppl_lines.get_figure()
ppl_fig.savefig('../logs/ppl_plot')



