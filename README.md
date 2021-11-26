# Neural-Natural-Logic
Implementation of the neural natural logic paper on natural language inference.

## Exploring End-to-End Differentiable Natural Logic Modeling (COLING 2020)

Our model combines Natural Logic from Stanford and the neural network. We squeeze intrepretability from the black box neural model by forcing it to learn and reason according to the natural logic framework. 
This research is in its early stage and we are still working to enhance it. We cleaned our experiment code and release the core in this repo. 
Please contact the first author feng.yufei@queensu.ca for more info




## Running instructions:
1. Download SNLI data, Glove, StanfordCoreNLP, please find the code for the exact path to put them in.
2. Run prepro code to get snli data, vocab, word embedding.
3. Run aligner (please use the esim checkpoint provided below in the link, if you found vocabulary mis-match, it is due to the fact that the tokenizer is in a different version, please train a new esim model with the code provided with the aligner checkpoint).
4. run train_aligned (checkpoints available below).
5. run explain.

## If you need checkpoints and the align model checkpoint:
https://drive.google.com/file/d/17xyD31Aq8XsVLBVKKn4RagJRmeDNlsQ_/view?usp=sharing
or
https://queensuca-my.sharepoint.com/personal/17yf48_queensu_ca/Documents/Attachments/checkpoints.zip
or
send me (feng.yufei@queensu.ca) an email
## Please cite the following paper if you find our code helpful:
```
@inproceedings{feng2020exploring,
  title={Exploring End-to-End Differentiable Natural Logic Modeling},
  author={Feng, Yufei, and Liu, Quan and Greenspan, Michael and Zhu, Xiaodan},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={1172--1185},
  year={2020}
}
```
