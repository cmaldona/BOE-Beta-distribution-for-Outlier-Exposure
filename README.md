# BOE-Beta-distribution-for-Outlier-Exposure

This repository contains the experimental setup and scripts for evaluating and benchmarking the proposed approach against the state-of-the-art (SOTA).  

## Repository Structure  

  - **Jupyter Notebook** (`note_main.ipynb`):  
    - Main environment for running and analyzing experiments.  
    - Provides flexibility for modifying and fine-tuning experimental parameters.  
    - Most SOTA comparisons were conducted within this notebook.  

- **Library Scripts:**  
  - `data_scripts.py`: Utility functions for dataset preprocessing and handling.  
  - `functions.py`: Core functions for experimental computations and evaluations.  

- **Dataset Tracking:**  
  - `generator_setups.py`: Logs and tracks dataset generation setups, ensuring reproducibility of experimental data.  

- **Experimental Runner:**  
  - `exp_v1.2.py`: Standalone script for executing experiments on a server.  
  - Allows batch processing of multiple experimental runs.  
  - Useful for systematically comparing the proposed method against baseline approaches.  

## Datasets

Our primary datasets are:

- **CLINC150**: A user intent classification dataset with 150 classes.
- **NewsCategory**: A dataset with headlines and short descriptions from a news website, comprising 42 classes. We only use the headlines.
- **TREC**: A question classification dataset used in its coarse version with 6 classes.
- **20NGs**: A newgroups classification dataset with large text and 20 classes.

These datasets consist of short texts, averaging no more than 30 words each.

Additionally, we use the following datasets for far-OOD (Out-of-Distribution) data:
- **YELP**: User reviews for products.
- **SST2**: Sentences from movie reviews.
- **SNLI**: Human-written sentences.
- **Reddit**: Reddit posts.
- **IMDB**: Movie reviews.

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
| NG20           | NG20-O1/Reddit       | NG20-O1/Reddit       | NG20-O2/IMDB       |

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

## Contact

For any inquiries, please contact [camilo.maldonado@sansano.usm.cl](mailto:camilo.maldonado@sansano.usm.cl).
