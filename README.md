# Neural-Natural-Logic
Implementation of the neural natural logic paper on natural language inference.
Our model combines Natural Logic from Stanford and the neural network. We squeeze intrepretability from the black box neural model by forcing it to learn and reason according to the natural logic framework. 
This research is in its early stage and we are still working to enhance it. We cleaned our experiment code and release the core part in this repo. 
Please contact feng.yufei@queensu.ca for more info




Running instructions:
Download SNLI data, Glove, StanfordCoreNLP, please find the code for the exact path to put them in.
Run prepro code to get snli data, vocab, word embedding.
Run aligner.
run train_aligned.
run explain.

