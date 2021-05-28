# Neural-Natural-Logic
Implementation of the neural natural logic paper on natural language inference.

## Exploring End-to-End Differentiable Natural Logic Modeling (COLING 2020)

Our model combines Natural Logic from Stanford and the neural network. We squeeze intrepretability from the black box neural model by forcing it to learn and reason according to the natural logic framework. 
This research is in its early stage and we are still working to enhance it. We cleaned our experiment code and release the core in this repo. 
Please contact feng.yufei@queensu.ca for more info




## Running instructions:
1. Download SNLI data, Glove, StanfordCoreNLP, please find the code for the exact path to put them in.
2. Run prepro code to get snli data, vocab, word embedding.
3. Run aligner.
4. run train_aligned.
5. run explain.

## Please cite the following paper if you find our code helpful:
```
@inproceedings{feng2020exploring,
  title={Exploring End-to-End Differentiable Natural Logic Modeling},
  author={Feng, Yufei, Zi'ou Zheng, and Liu, Quan and Greenspan, Michael and Zhu, Xiaodan},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={1172--1185},
  year={2020}
}
```
