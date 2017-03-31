import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
        both lists are ordered by the test set word_id
        probabilities is a list of dictionaries where each key a word and value is Log Liklihood
            [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             ]
        guesses is a list of the best guess words ordered by the test set word_id
            ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    #Initialize the probabilities list
    probabilities = []
    
    # loop through every word_id in the test set to compute dictionary of logL
    for word_id in test_set.get_all_Xlengths().keys():
        X, lengths = test_set.get_item_Xlengths(word_id)
        logL_dict = dict() # initialize the dict of logL
        
        # loop through each model in the trained models dict
        for word in models.keys():
            try:
                logL = models[word].score(X, lengths) 
                logL_dict[word] = logL
            except:
                logL_dict[word] = None
        probabilities.append(logL_dict)
        
    # Initialize the guesses list
    guesses = []
    
    # loop through every dictonary in probablities, find the word with the maximum logL and append it to the guesses
    for dictionary in probabilities:
        max_logL = -float('inf')
        guess = None
        for word in dictionary.keys():
            if dictionary[word] is not None and dictionary[word] > max_logL:
                max_logL = dictionary[word]
                guess = word
        guesses.append(guess)
        
    return probabilities, guesses
                
        
