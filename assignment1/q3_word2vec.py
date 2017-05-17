#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x_dimension = x.shape[0]
    x_norm = np.sqrt(np.sum(x**2, axis=1)).reshape((x_dimension,1))
    x = x / x_norm
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "--- Testing normalizeRows... ---"
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print "--- End of normalizeRows test ---\n"


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    probs = softmax(np.dot(predicted, outputVectors.T))
    cost = -np.log(probs[target])

    scores_diff = probs
    scores_diff[target] -= 1
    gradPred = np.dot(scores_diff, outputVectors) # derived in q3a. Note: gradPred should have the same dim. as the predicted vector since we are differentiating w.r.t vc.
    
    dim_m, dim_n = outputVectors.shape
    grad = np.dot(scores_diff.reshape((dim_m, 1)), predicted.reshape((1, dim_n))) # derived in q3b. Note: grad should have the same dim. as the outputVectors since we are differentiating w.r.t uw.
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = []
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    # COMPUTING SOFTMAX COST
    target_row_vector = outputVectors[target]
    sigmoid_uoT_vc = sigmoid(np.dot(target_row_vector, predicted)) # returns one value
    
    neg_sample_vectors = outputVectors[indices]
    sigmoid_ukT_vc = sigmoid(-np.dot(neg_sample_vectors, predicted))
    
    cost = -np.log(sigmoid_uoT_vc) - np.sum(np.log(sigmoid_ukT_vc))
    # END OF COMPUTING SOFTMAX COST

    # COMPUTING GRADIENT FOR INPUT AND OUTPUT VECTORS
    sigmoid_minus_one = sigmoid_uoT_vc - 1
    sigmoid_neg_minus_one = sigmoid_ukT_vc - 1
    
    gradPred_first_term = sigmoid_minus_one * target_row_vector
    gradPred_second_term = np.dot(sigmoid_neg_minus_one, neg_sample_vectors)
    gradPred = gradPred_first_term - gradPred_second_term # derived in q3c. Note: gradPred should have the same dim. as the predicted vector since we are differentiating w.r.t vc.

    grad = np.zeros(outputVectors.shape)
    grad[target] = np.dot(sigmoid_minus_one,predicted)
    for index, pos in enumerate(indices):
    	grad[pos] += -(sigmoid_neg_minus_one)[index] * predicted
    # COMPUTING GRADIENT FOR INPUT AND OUTPUT VECTORS
    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(centerWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### DEBUGGING PURPOSE
    # print "Center word: ", centerWord
    # print "Context size: ", C
    # print "Context words: ", contextWords
    # print "Tokens with index: ", tokens
    # print "InputVectors: ", inputVectors
    # print "CenterWord row vector: ", inputVectors[tokens[centerWord]]
    # print "Output vectors: ", outputVectors
    # print "Dataset: ", dataset
    ### END OF DEBUGGING

    ### YOUR CODE HERE
    centerWord_pos = tokens[centerWord]
    centerWord_vector = inputVectors[centerWord_pos]
    
    for context_word in contextWords:
    	target = tokens[context_word]
    	softmax_cost, grad_pred, grad_out = word2vecCostAndGradient(centerWord_vector, target, outputVectors, dataset)
    	cost += softmax_cost
    	gradIn[centerWord_pos] += grad_pred
    	gradOut += grad_out
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(centerWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    indices = [tokens[w] for w in contextWords]  
    context_words_vector = inputVectors[indices]
    context_words_vector = np.sum(context_words_vector, axis=0)
    target = tokens[centerWord]    

    softmax_cost, grad_in, grad_out = word2vecCostAndGradient(context_words_vector, target, outputVectors, dataset)
    # The following for loop seems unavoidable. We couldn't perform this operation: 
    #    ```gradIn[indices] += grad_in``` cuz numpy doesn't seem to update (add) rows of elements in place.
    for index in indices:
        gradIn[index] += grad_in

    ### END YOUR CODE
    cost = softmax_cost
    gradOut = grad_out

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()  # return a new type object
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()