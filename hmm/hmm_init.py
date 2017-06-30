import codecs, collections, sys
import numpy as np
import glob, os, nltk, itertools as it

# def extract_test_training():
#     total_characters=0
#     for filename in glob.glob('originals/*.txt'):
#         with codecs.open(filename, 'r', 'utf-8-sig') as tweets:
#             with codecs.open('training/knowledge_base.txt', 'w', 'utf-8-sig') as know:
#                 with codecs.open('test/test_of_remains.txt', 'w', 'utf-8-sig') as test:
#                     size = (os.stat(filename).st_size/100.0)*80
#                     for line in tweets:
#                         for character in line:
#                             if total_characters<size:
#                                 total_characters +=len(character)
#                                 know.write(unicode(character))
#                             else:
#                                 test.write(unicode(character))


def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v))
        for k, v in dictionary.items())

def convert(data):
    if isinstance(data, basestring):
        return data.encode('utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data
def count_character():
    number_of_characters = {}
    total_characters = 0.0
    for filename in glob.glob('training/Pontifex_knowledge_base.txt'):
        with codecs.open(filename, 'r', 'utf-8-sig') as tweets:
            for line in tweets:
                line = convert(line)
                bigrams = list(nltk.bigrams(line))
                for bi in bigrams:
                    if bi[0]!= '\n' or bi[1]!= '\n':
                        if bi in number_of_characters.keys():
                            number_of_characters[bi]+=1
                        else:
                            number_of_characters[bi]=1
                        total_characters +=1
    #for k in number_of_characters:
    #    print repr(k), number_of_characters[k]
    return number_of_characters, total_characters

def prior_probability():
    state_list = states()
    length_state = len(state_list)
    prior_prob= np.full(length_state, 0.0)
    numb_of_char, total = count_character()
    s = 0.0

    #Transform in frequences numb_of_char
    for k,v in numb_of_char.iteritems():
        prior_prob[state_list.index(k)]=numb_of_char[k]/total
    #Verify if prior_probability sums to 1
    #for i in range(0,length_state):
        #s+=prior_probability[i]
    #print s
    return np.matrix(prior_prob)
def states():
    numb_of_char, total = count_character()
    list_of_states = numb_of_char.keys()
    #print list_of_states
    return list_of_states

def calc_probabilities_transictions(row):
    sum = np.sum(row)
    return row / sum

def transition_model():
    numb_of_char , total= count_character()
    list_of_states = states()
    n_states = len(list_of_states)
    transitions = np.asmatrix(np.full((n_states, n_states), 1.0/sys.maxint))

    for filename in glob.glob('training/Pontifex_knowledge_base.txt'):
        print "Training of "+filename+"..."
        with codecs.open(filename, 'r', 'utf-8-sig') as tweets:
            for line in tweets:
                line = convert(line)
                bigrams = list(nltk.bigrams(line))
                for i in range(0, len(bigrams)-1):
                    if bigrams[i]!='\n' and bigrams[i+1]!='\n':
                        row_index = list_of_states.index(bigrams[i])
                        col_index = list_of_states.index(bigrams[i+1])
                        transitions[row_index,col_index]+=1
    transitions = np.apply_along_axis( calc_probabilities_transictions, axis=1, arr=transitions )
    #'print' transitions
    return transitions

def observation():
    observations = states()
    return observations

def emission_probability(adj_list):
    #print adj_list
    obs = observation()
    n_obs = len(obs)
    #print obs
    epsilon = 1.0/1000
    emissions = np.asmatrix(np.full((n_obs, n_obs), 0.0))
    likelihood = 0.8
    for k,v in adj_list.iteritems():
        if k in obs:
            state = obs.index(k)
            emissions[state, state] += likelihood
            #print ('v', ' ') in obs
            c = []
            for couple in v:
                if couple in obs and couple != k:
                    c.append(couple)
            weight = 0.2/len(c)
            #print c
            for couple in c:
                emissions[state, obs.index(couple)] = weight

            # epsilon=0.02/(len(obs)-len(c)-1)
            # for j in range(0,len(obs)):
            #     if emissions[state,j]==0:
            #             emissions[state,j] = epsilon

    print np.sum(emissions, axis=1)
    return emissions
