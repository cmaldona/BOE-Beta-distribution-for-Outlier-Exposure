# BOE-Beta-distribution-for-Outlier-Exposure

## Overview

This repository contains the initial version of our research project conducted in Google Colab. It includes two primary Jupyter notebooks:

1. **State-of-the-Art Experiments**: This notebook replicates experiments from existing literature.
2. **Proposed Method Experiments**: This notebook demonstrates our proposed method and its results.

## Notebooks

- `state_of_the_art_experiments.ipynb`: Contains the implementation and results of experiments based on MSP, ViM and OE with uniform distribution (the baseline) methods.  
- `proposed_method_experiments.ipynb`: Contains the implementation and results of our proposed method with Beta distribution for the regularization.

## Datasets

Our primary datasets are:

- **CLINC150**: A user intent classification dataset with 150 classes.
- **NewsCategory**: A dataset with headlines and short descriptions from a news website, comprising 42 classes. We only use the headlines.
- **TREC**: A question classification dataset used in its coarse version with 6 classes.

These datasets consist of short texts, averaging no more than 30 words each.

Additionally, we use the following datasets for far-OOD (Out-of-Distribution) data:
- **YELP**: User reviews for products.
- **SST2**: Sentences from movie reviews.
- **SNLI**: Human-written sentences.

For near-OOD data, only CLINC150 includes labeled OOD data. For the other datasets, we exclude certain classes to create near-OOD data, ensuring no overlap between near-OOD classes in training/validation and testing stages.

We distinguish between different partitions as follows:
- **TREC-I**: The most repeated classes from the original TREC dataset (ID).
- **TREC-O1** and **TREC-O2**: Disjoint sets of classes defined as OOD.

## Experimental Setup

Each value in the table indicates the types of near-OOD/far-OOD data respectively. The first column represents the main dataset as ID.

|                | Train              | Validation         | Testing            |
|----------------|--------------------|--------------------|--------------------|
| CLINC150       | Clinc150/SST2      | Clinc150/YELP      | Clinc150/NC        |
| News Category  | NC-O1/SNLI         | NC-O1/YELP         | NC-O2/SST2         |
| TREC           | TREC-O1/SST2       | TREC-O1/YELP       | TREC-O2/SNLI       |

*Table \ref{tab:setups}: Experimental setups with near-OOD/far-OOD data.*

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Google Colab (optional)

### Running the Notebooks

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/yourusername/repository-name.git](https://github.com/cmaldona/BOE-Beta-distribution-for-Outlier-Exposure.git)
   cd BOE-Beta-distribution-for-Outlier-Exposure
   ```
2. **Open the jupyter notebooks in Jupyter or Colab**.

## Environment

The experiments were conducted in Google Colab using T4 GPUs.

## Contact

For any inquiries, please contact [camilo.maldonado@sansano.usm.cl](mailto:camilo.maldonado@sansano.usm.cl).
