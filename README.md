
# Lung CT Pseudo-Group Reconstruction

This repository provides the implementation of the similarity-aware pseudo-group reconstruction method for lung CT images.

The method reconstructs groups of visually related CT images using DINOv2 patch-feature similarity, without relying on patient identifiers, acquisition order, or slice adjacency information.

## Method Overview

The reconstruction pipeline consists of five stages:

1. **Local Feature Similarity:** DINOv2 patch feature extraction, median cosine similarity, and spatial AUC.
2. **Strict Seed Construction:** Initial group construction using strict similarity criteria and mutual nearest neighbors.
3. **Preliminary Singleton Expansion:** Evidence-based attachment of eligible singleton images to existing seed groups.
4. **Adaptive Edge Selection:** Adaptive threshold estimation and selection of candidate relationships.
5. **Conservative Group Refinement:** Cohesion-constrained greedy merging of candidate groups.

Detailed methodological definitions and reconstruction parameters are provided in [ALGORITHM.md](ALGORITHM.md).

## Installation

Python 3.10 or later is required.

```bash
pip install -e .
```

## Citation

Citation information will be provided upon publication.
