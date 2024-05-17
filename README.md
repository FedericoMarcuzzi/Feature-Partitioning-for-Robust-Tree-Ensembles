# Feature Partitioned Forest

This repository contains the implementation of *Feature Partitioned Forest* proposed by **Calzavara *et. al.*** in their research paper titled [<em>Feature partitioning for robust tree ensembles and their certification in adversarial scenarios</em>](https://jis-eurasipjournals.springeropen.com/articles/10.1186/s13635-021-00127-0).

## Abstract
---
Machine learning algorithms, however effective, are known to be vulnerable in adversarial scenarios where a malicious user may inject manipulated instances. In this work, we focus on evasion attacks, where a model is trained in a safe environment and exposed to attacks at inference time. The attacker aims at finding a perturbation of an instance that changes the model outcome.We propose a model-agnostic strategy that builds a robust ensemble by training its basic models on feature-based partitions of the given dataset. Our algorithm guarantees that the majority of the models in the ensemble cannot be affected by the attacker. We apply the proposed strategy to decision tree ensembles, and we also propose an approximate certification method for tree ensembles that efficiently provides a lower bound of the accuracy of a forest in the presence of attacks on a given dataset avoiding the costly computation of evasion attacks.Experimental evaluation on publicly available datasets shows that the proposed feature partitioning strategy provides a significant accuracy improvement with respect to competitor algorithms and that the proposed certification method allows ones to accurately estimate the effectiveness of a classifier where the brute-force approach would be unfeasible.

## Requirments
---
In order to train [Robust Trees](http://proceedings.mlr.press/v97/chen19m.html) we used the implementation provided in the following repository: [treant](https://github.com/gtolomei/treant). 


## Credit
---
If you use this implementation in your work, please add a reference/citation to our paper. You can use the following BibTeX entry:

```
@article{DBLP:journals/ejisec/CalzavaraLMO21,
  author       = {Stefano Calzavara and
                  Claudio Lucchese and
                  Federico Marcuzzi and
                  Salvatore Orlando},
  title        = {Feature partitioning for robust tree ensembles and their certification
                  in adversarial scenarios},
  journal      = {{EURASIP} J. Inf. Secur.},
  volume       = {2021},
  number       = {1},
  pages        = {12},
  year         = {2021},
  url          = {https://doi.org/10.1186/s13635-021-00127-0},
  doi          = {10.1186/S13635-021-00127-0},
  timestamp    = {Sat, 08 Jan 2022 02:23:43 +0100},
  biburl       = {https://dblp.org/rec/journals/ejisec/CalzavaraLMO21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
