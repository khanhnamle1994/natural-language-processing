import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y
    ### END YOUR CODE
    return x

def l1_normalize_rows(x):
    """ l1 row normalization function """
    y = None

    y = np.sum(x,axis=1,keepdims=True)
    x /= y
    return x

def l2_normalize_rows(x):
    """ l1 row normalization function """
    y = None

    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    N, D     = outputVectors.shape

    r    = predicted
    prob = softmax(r.dot(outputVectors.T))
    cost = -np.log(prob[target])

    dx   = prob
    dx[target] -= 1.

    grad     = dx.reshape((N,1)) * r.reshape((1,D))
    gradPred = (dx.reshape((1,N)).dot(outputVectors)).flatten()
    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    N, D     = outputVectors.shape

    cost     = 0
    gradPred = np.zeros_like(predicted)
    grad     = np.zeros_like(outputVectors)

    #negative_samples = np.array([dataset.sampleTokenIdx() for i in range(K)], dtype='int64')
    negative_samples = []
    for k in range(K):
        new_idx = dataset.sampleTokenIdx()
        while new_idx == target:
            new_idx = dataset.sampleTokenIdx()
        negative_samples += [new_idx]
    indices = [target]
    indices += negative_samples

    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outputVectors[indices]

    z        = np.dot(vecs, predicted) * labels
    probs    = sigmoid(z)
    cost     = - np.sum(np.log(probs))

    dx = labels * (probs - 1)
    gradPred = dx.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = dx.reshape((K+1,1)).dot(predicted.reshape(1,predicted.shape[0]))

    for k in range(K+1):
        grad[indices[k]] += gradtemp[k,:]

#     t = sigmoid(predicted.dot(outputVectors[target,:]))
#     cost = -np.log(t)
#     delta = t - 1
#     gradPred += delta * outputVectors[target, :]
#     grad[target, :] += delta * predicted
#     for k in xrange(K):
#         idx = dataset.sampleTokenIdx()
#         t = sigmoid(-predicted.dot(outputVectors[idx,:]))
#         cost += -np.log(t)
#         delta = 1 - t
#         gradPred += delta * outputVectors[idx, :]
#         grad[idx, :] += delta * predicted

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    cost    = 0
    gradIn  = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)

    c_idx     = tokens[currentWord]
    predicted = inputVectors[c_idx, :]

    #__TODO__: can be switched to vectorized;
    # target (need to know shape; think its just a number)
    # hence target = np.zeros(len(contextWords))?
    # can add a newaxis(?) to allow for broadcasting
    for j in contextWords:
        target = tokens[j]
        c_cost, c_gradPred, c_grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += c_cost

        gradIn[c_idx,:] += c_gradPred
        gradOut         += c_grad

    #Vectorizing like this will be slow (size of outputVectors.shape[0] is equal # words)
    # samples     = np.array([tokens[j] for j in contextWords])
    # bin_samples = np.bincount(samples, minlength=outputVectors.shape[0])
    # print(np.nonzero(bin_samples))
    # for i in np.nonzero(bin_samples)[0]:
    #     word2vecCostAndGradient(predicted, i, outputVectors, dataset)
    # A = np.diag(bin_samples)

    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    c_idx     = tokens[currentWord]
    onehot    = np.zeros((2*C, len(tokens)))

    for i, word in enumerate(contextWords):
        onehot[i, tokens[word]] += 1.

    # print(onehot)
    d = np.dot(onehot, inputVectors)
    predicted = 0.5 / C * np.sum(d, axis=0)

    cost, gradPred, gradOut = word2vecCostAndGradient(predicted, c_idx, outputVectors, dataset)

    gradIn = np.zeros(inputVectors.shape)
    for word in contextWords:
        gradIn[tokens[word]] += 0.5 / C * gradPred
    ### END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
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

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
