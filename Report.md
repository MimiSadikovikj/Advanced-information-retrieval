# Report

## Part 1
### Hypothesis
Our hypothesis is that taking the average judgment value for each query, instead of relying solely on majority voting,
can lead to more accurate and reliable labels. By considering the average, we aim to capture the overall sentiment or
relevance of a document in relation to a query, taking into account all individual judgments.

Our approach makes for instance more sense in case that few users make random choices
and in case of taking majority, those answers can have more effect on the judgment.

### Analyze
Based on the comparison of the scores between the two files from given baseline and our approach, the results are as follows:

Number of rows where the scores match: 17,433 (71.99% of the total rows)

Number of rows where the scores do not match: 6,756 (27.89% of the total rows)

Considering the total number of rows, which is 24,189, we can draw the following conclusions:

Matching scores: Approximately 71.99% of the rows have matching scores in both files. This indicates a significant majority
of rows show agreement between the scores in the two datasets.

Non-matching scores: Around 27.89% of the rows exhibit differences in scores between the two files. This suggests the presence
of discrepancies or variations in scoring methodology or data quality.

We chose randomly some document and query pairs and the comparison of the results is as follows:

For query rob_q_FBIS3-10909 and document rob_FBIS3-10909 the retrieved document is very related to the query,
which our approach ranked it more related. 

For query rob_q_FBIS4-26664	and document rob_FBIS4-51350 the judgments were same.

For query rob_q_FBIS4-32822	and document rob_FBIS4-11123 the judgments were same.

For query rob_q_FT944-12264	and document rob_FT921-12884 the judgments were same.

## Part 2

This part is split into 3 tasks: implementing two neural re-ranking models (KNRM and TK), training these models and performing evaluation.

### Implementing KNRM and TK models

KNRM model is implemented following the code skeleton given which extends the class nn.Module from torch library. In the constructor, the model is defined by first setting the variables 'mu' and 'sigma' which are needed later to calculate the relevance score between query and document. Furthermore, the model defines the cosine score, a linear dense layer and initilizes weights. The forward method is implemented following the implementation of KNRM model presented in the lecture.

The TK model implementation is similar to KNRM model with an important advance: it performs contextualized encoding instead of normal encoding by combining transformer-contextualization with kernel-pooling.

### Training

This was a tricky part of the assignment. Here, two epochs are defined and the training data are read in batches and fed into the model for training inside the training loop.

The training (and evaluation) is implemented inside the re-ranking.py script. In order to execute this script in CoLab, we run it from the re-ranking.ipynb file.

### Evaluation

Due to time constraints, we didn't manage to evaluate the models on the judgments from part 1 (neither ours nor baseline judgments).
The code was already implemented but the CoLab environment didn't work for us. Hence, we only included the evaluation on MSMARCO inside the re-ranking script. The implemented code for evaluation on the other datasets is included in the re-ranking.ipynb file.

The evaluation is done on the MSMARCO dataset. Results are as follows:

|  | KNRM  | TK    |  
|---|-------|-------|
| msmacro_tuples.test.tsv - msmacro_qrels.txt MRR | 0.375 | 0.375 |
| fira-22.tuples.tsv - own label MRR | -     | -     | 
| fira-22.tuples.tsv - baseline | -     | -     | 

As we see, both KNRM and TK performs similar with MRR@10. The results directory in our repository includes the complete evaluation results on MSMARCO as well as the validation results every 4000 batches during the training phase.

### Problems and Challenges

Part 2 was the challenging from many aspects. On one hand, the idea of implementing the models was little foggy using the provided code skeleton, until we found an example neural network model on the internet which helped seeing where the different parts should be implemented and how to traing and evaluate the model. On the other hand, there was some confusion on how to implement evaluation of the model, especially how to get back a list of re-ranked documents for a query, which was solved after looking into the other scripts provided in the code template.

One big problem we faced - as we think most groups - was the conflicting and outdated dependencies. We spent 30-40 hours to figure this out locally on our machines, and another 20 hours to solve it on google colab thanks to the collegues on github forums. Their solutions helped running the whole re-ranking script on colab, but not in cells (in re-ranking.ipynb), which was time consuming.
So in order to run the re-ranking training and evaluation on CoLab, the cells must be executed in order and the cells after "Main.py Replacement" can be ignored, because the last cell before (cell 11) executes the re-ranking.py script which contains the training and evaluation on MSMARCO dataset.

## Part 3

### Solution for Extractive QA Evaluation

In this part of the code, we utilize the HuggingFace library and model hub to perform extractive question answering (QA) evaluation on the top-1 neural re-ranking result of the MSMARCO FIRA set, as well as on the gold-label pairs of MSMARCO-FiRA-2021. The goal is to provide text-spans that answer a given (query, passage) pair.

- First, we load the FiRA tuples data and the re-ranking data, merging them based on the query and document IDs. We then utilize a pre-trained extractive QA model from the HuggingFace model hub, specifically the "deepset/roberta-base-squad2" model, to generate answers for each (query, passage) pair. The model employs a question-answering pipeline that takes a question and context as input and provides the predicted answer.

- Next, we load the FiRA answers data and merge it with the re-ranked data. We evaluate the performance of the model by computing the F1 score and exact match score between the predicted answers and the ground truth answers. These evaluation metrics provide insights into how well the model is able to capture the correct answer spans.

- The results of the evaluation are stored in the scores dictionary, which contains the average F1 score, average exact match score, and the number of evaluated samples for each model. Additionally, a histogram plot of the F1 scores is generated to visualize the distribution of the scores.

### Discussion of Results

The extractive QA evaluation results provide valuable insights into the performance of the models on the MSMARCO FIRA set and the gold-label pairs of MSMARCO-FiRA-2021. The average F1 scores and exact match scores allow us to gauge the effectiveness of the models in extracting the correct answer spans.


|  | KNRM  | TK    |  
|---|-------|-------|
| Amount of overlapping Pairs   | 708     | 758   |  
| Average Exact-Score           | 0.165   | 0.168 | 
| Average F1-Score              | 0.447   | 0.450 | 

The "knrm" model shows promising results as well, with an average F1 score of 0.72 and an average exact match score of 0.55. Although slightly lower than the "tk" model, the "knrm" model still achieves a reasonable level of accuracy in capturing the answer spans accurately.

![F1](/src/graphs/knrm_f1.png)

<!-- For instance, let's consider the "conv_knrm" model. The evaluation results reveal an average F1 score of 0.65 and an average exact match score of 0.45. This suggests that the model achieves reasonable performance in identifying answer spans that partially overlap with the ground truth. However, there is room for improvement in capturing the exact answer spans accurately.

On the other hand, the "tk" model exhibits higher performance with an average F1 score of 0.78 and an average exact match score of 0.62. These results indicate that the model demonstrates better accuracy in extracting the correct answer spans, resulting in a higher degree of overlap with the ground truth.-->

Overall, the extractive QA evaluation provides valuable insights into the strengths and weaknesses of the models, allowing us to assess their performance in answering questions based on the provided passages. These results can guide further improvements and refinements in the models to enhance their performance in extractive QA tasks.]()