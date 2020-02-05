# Classification of Biomedical Questions from the BioASQ Challenge - Task 6b, extended with co-training from Quora questions

According to its original formulation, the BioASQ Challenge - Task 6b is a Question-Answering competition in the biomedical domain. The organizers provided a set of questions, divided in two sets, training and test, with the latter being labeled with a specific question type (or class), as well as an answer. In this work two tasks are solved:

- Task 1: Development of a classifier of biomedical questions from the challenge, into 4 question types: factoid, summary, list and yes/no.

- Task 2: Co-training with an extensive broad-domain data set of 404K Quora question pairs to improve the classification accuracy, since the data of Task 1 is comprised only of a small number of questions (2252). Nevertheless, the Quora questions need to be filtered and classified first, before being usable in the model.

