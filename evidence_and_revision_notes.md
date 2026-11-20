# Evidence and proposed manuscript additions

The following describes a fresh run of the repository code on the locally cached 20 Newsgroups training subset. The run used scikit-learn 1.6.1, 30 K-Means seeds (0–29), and the same TF-IDF, SVD and consensus settings as the current source. It is one ensemble, not an independent repeated-ensemble study. The script and full per-run metrics accompany this note.

## Dataset and protocol

The four selected categories contain 2,338 documents, with class counts of 584, 597, 593 and 564 in the dataset's target-name order. TF-IDF produced 7,252 features before reduction to 200 SVD dimensions. Headers were retained; footers and quotes were removed. Four clusters were specified using the known number of categories, so labels informed the choice of cluster count even though they were not used to fit the models. K-Means uses the installed scikit-learn default for n_init; the n_init setting in config.py is not passed to KMeans. This dependency must be pinned or recorded to support reproduction.

## Additional external evaluation

| Method | ARI | NMI | AMI |
|---|---:|---:|---:|
| K-Means, mean ± population SD across 30 seeds | 0.206 ± 0.168 | 0.303 ± 0.168 | 0.302 ± 0.168 |
| Best K-Means by ARI | 0.556 | 0.572 | 0.571 |
| Central representative | 0.180 | 0.346 | 0.345 |
| Co-association and average linkage | 0.579 | 0.582 | 0.582 |

NMI and AMI measure information shared with the categories; AMI adjusts for chance. The best K-Means row is selected by ARI and uses external labels, making it an oracle benchmark rather than an unsupervised model-selection rule. The consensus improves on the base-run mean by 0.372 ARI points and the best observed base run by 0.023 points. These are descriptive, dependent comparisons, not significance tests. The 30 base runs share the same dataset and representation. A 30-seed standard deviation describes initialization variability; it is not a confidence interval for generalization. The supplementary figure displays each base score and the two consensus scores.

The reproduced representative ARI (0.180, seed 21) differs from the manuscript's 0.184. Both are members of this same ensemble; investigate the pairwise-similarity totals, ties, software versions and exact saved run before replacing the published value. The co-association result matches the manuscript to the reported precision. The code's dense float64 co-association matrix alone requires about 41.7 MiB for 2,338 documents; hierarchical linkage and other intermediate arrays require additional memory. Runtime from a single local run is machine-dependent and is not a scalability benchmark.

## Survey synthesis for inclusion

A useful comparison is the object each strategy aggregates: representative selection compares whole partitions and returns one input partition; voting requires a correspondence between cluster labels; co-association aggregates observation pairs and is invariant to label names; graph methods turn ensemble relations into a partitioning problem; weighted and selective methods modify which evidence contributes. This distinction connects the survey to the experiment: the representative can only return a base partition, while pairwise aggregation can produce a new partition. The case study compares only two consensus functions on one fixed K-Means ensemble and does not test whether this distinction holds across datasets, representations, or other clustering algorithms. The existing references [3], [4], [6], [13], [16] and [17] provide starting points for source verification before incorporating this synthesis.

## Gaps before claiming comprehensive evaluation

Run multiple independent ensembles with disjoint seed sets and report their distribution; compare several datasets and baselines; vary ensemble size and representation; report runtime and peak memory under documented hardware; and examine class-level confusion and cluster sizes. Evaluate sensitivity to retained headers, which may contain category-identifying information. These experiments have not been performed here. Confirm every literature citation against its original publication and disclose the survey's search strategy and selection criteria before presenting it as systematic.
