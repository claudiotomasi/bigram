from hmm import hmm_init
from utility import utility
import hidden_markov as hmlib
import codecs, pickle, argparse, glob
from sklearn import metrics
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Training or not', action='store_true')
args = parser.parse_args()
if args.train:
    print "With Training"
    prior = hmm_init.prior_probability()
    transitions = hmm_init.transition_model()
    states=hmm_init.states()
    possible_obs = hmm_init.observation()
    emissions = hmm_init.emission_probability(utility.adjacents_bigrams())
    model = hmlib.hmm(states, possible_obs, prior, transitions, emissions)
    afile = open('training/model', 'wb')
    pickle.dump(model, afile)
    afile.close()
    print "End of training\n"

else:
    print "Without Training\n"
    afile = open('training/model', 'rb')
    model = pickle.load(afile)
    afile.close()
#print len(states),len(emissions)


states = model.states

for filename in glob.glob('test/Pontifex_test_of_remains.txt'):
    tweet = ""
    with codecs.open(filename , "r", 'utf-8-sig') as f:
        print "Start prediction of "+filename+"..."
        for tweets in f:
            tweet += hmm_init.convert(tweets)
    #elimino newline
    tweet = tweet.replace('\n','')
    #separo per frasi
    list_of_words = tweet.split('.')
    correct = ""
    pred = []
    for word in list_of_words:
        #se frase non vuota aggiungo un punto alla fine
        if word!= '':
            word+='. '
        obs = list(word)
        #print obs
        if obs!=[]:
            # Rimuovo spazi all'inizio
            if obs[0]==' ':
                obs = obs[1:]
            if obs[0]!='\n':
                bigrams = list(nltk.bigrams(''.join(obs)))
                pred += model.viterbi(bigrams)
    #trasformo lista predizioni in stringa
    print pred
    for l in pred:
        correct += ''.join(l)

    with codecs.open(filename+'_correct', 'w', 'utf-8-sig') as f:
        correct = unicode(correct)
        f.write(correct)
    print "End prediction of "+filename+"\n"



#confusion_matrix(real, pred, states, sample_weight=None)
#Si potrebbe calcolare viterbi sulle parole e non sulle frasi
