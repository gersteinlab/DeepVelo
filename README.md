# DeepVelo

Single-cell Transcriptomic Deep Velocity Field Learning with Neural Ordinary Differential Equations

## Note
This is currently an updating repository. We are currently working on packaging the scripts into a Python package. For publication purposes, we have temporarily deposited the raw notebooks used for the analysis here.

## Abstract

Recent advances in single-cell RNA sequencing technology provided unprecedented opportunities to simultaneously measure the gene expression profile and the transcriptional velocity of individual cells, enabling us to sample gene regulatory network dynamics along developmental trajectories. However, traditional methods have been challenged in offering a fundamental and quantitative explanation of the dynamics as differential equations due to the high dimensionality, sparsity, and complex gene interactions. Here, we present DeepVelo, a neural-network-based ordinary differential equation that can learn to model non-linear, high-dimensional single-cell transcriptome dynamics and describe gene expression changes of individual cells across time. We applied DeepVelo on multiple published datasets from different technical platforms and demonstrate its utility to 1) formulate transcriptome dynamics of different timescales; 2) measure the instability of individual cell states; and 3) identify developmental driver genes upstream of the signaling cascade. Benchmarking with state-of-the-art methods shows that DeepVelo can improve velocity field representation accuracy by at least 50% in out-of-sample cells. Further, our perturbation studies revealed that single-cell dynamical systems may exhibit properties similar to chaotic systems. In summary, DeepVelo allows for the data-driven discovery of differential equations that delineate single-cell transcriptome dynamics. 

### Dependencies

The python packages needed for the analysis are in the requirement.txt file. They can be installed by executing:

```
pip install -r requirements.txt
```

### Original Paper
[https://www.biorxiv.org/content/10.1101/2022.02.15.480564v2](https://www.biorxiv.org/content/10.1101/2022.02.15.480564v2)
