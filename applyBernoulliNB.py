from naivebayes import trainBernoulli

train = [
    ('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg')
]
test = [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

def applyBernoulliNB(trainSet, testSentence):
    trainBernoulliNB = trainBernoulli(trainSet)
    vocab, priorPos, priorNeg, posProb, negProb = trainBernoulliNB.trainAlgo()
    for sentences in testSentence:
        score = {}
        score["pos"] = priorPos
        score["neg"] = priorNeg

        for word in vocab:
            if word in sentences:
                score["pos"] *= posProb[word]
                score["neg"] *= negProb[word]
            else:
                score["pos"] *= (1 - posProb[word])
                score["neg"] *= (1 - negProb[word])
        print ("{} {}".format(max(score, key=score.get), score[max(score, key=score.get)]))

testSet = [sentence[0] for sentence in test]
applyBernoulliNB(train, testSet)