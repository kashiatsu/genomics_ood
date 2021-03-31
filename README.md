# Bacteria Genomics OOD dataset

This dataset implements a PyTorch dataset for the Genomics OOD dataset proposed in
> J. Ren et al., “Likelihood Ratios for Out-of-Distribution Detection,” arXiv:1906.02845 [cs, stat], Available: http://arxiv.org/abs/1906.02845.

The dataset contains for each input sample
 - A sequence of 250 integers, where each number is from {0, 1, 2, 3} indicating {A, C, G, T}. 
 - A class label, range from 0 to 129 for the bacteria class.
 - A a string notating where the sequence comes from.

In total there a 5 splits: Train, Validation, Test split with 10 in-distribution classes and a valdidation out-of-distribution dataset, as well as a out-of-distribution test set with 60 classes each.

The dataset with generated indices can be downloaded via [Kaggle](https://www.kaggle.com/svenel/genomics-ood).

### Attribution
The original dataset was released by
> Jie Ren, Google Research, 05/23/2019, jjren@google.com

Following CC BY 4.0 International [license](https://creativecommons.org/licenses/by/4.0/legalcode), this is released and distributed under the CC BY 4.0 license. 
The original dataset can be found [here](https://github.com/google-research/google-research/tree/master/genomics_ood).
