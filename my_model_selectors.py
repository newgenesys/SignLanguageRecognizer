import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, words: dict, hwords: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=None, verbose=False):
        self.words = words
        self.hwords = hwords
        self.sequences = words[this_word]
        self.X, self.lengths = hwords[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # compute the sample size N
        N = len(self.sequences)
        
        # compute the number of features
        n_features = self.X.shape[1]
        
        # get the minimum number of states amaong the samples
        max_n_components = min(self.lengths)
        
        #the self.max_n_component shouldn't be greater than the minimum number of states among the samples
        if max_n_components < self.max_n_components:
            self.max_n_components = max_n_components
            
        # initialize the dictionary that stores BIC value of different n_states choices
        # key: n_states, value: BIC value
        n_states_dict = dict()
        for n_states in range(self.min_n_components, self.max_n_components+1):
            n_states_dict[n_states] = 0
                
        # fit models of different n_states ranging from self.min_n_components to self.max_n_components
        for n_states in range(self.min_n_components, self.max_n_components+1):
            hmm_model = self.base_model(n_states) # fit the model
                
            #update the dictionary
            if hmm_model is not None:
                try:
                    logL = hmm_model.score(self.X, self.lengths)
                    
                    # compute the number of parameters (p) in the model
                    # for each state, each feature has two parameters to estimate, they are mean and std
                    # also, for each state, the transition probability is a parameter to estimate
                    p = n_states * (n_features * 2 + 1)
                    
                    BIC = -2 * logL + p * np.log(N) # compute BIC
                    n_states_dict[n_states] += BIC
                except: # in the case that the model doesn't have viable BIC, we don't update the dictionary
                    n_states_dict[n_states] += 0

        #initialize the optimal_n_state
        optimal_n_states = self.min_n_components
        while optimal_n_states <= self.max_n_components and n_states_dict[optimal_n_states] == 0:
            optimal_n_states += 1   
            
        # loop through the dictionary to find the optimal_n_state that has the smallest BIC
        if optimal_n_states <= self.max_n_components:
            for n_states in n_states_dict.keys():
                if n_states_dict[n_states] != 0 and n_states_dict[n_states] < n_states_dict[optimal_n_states]:
                    optimal_n_states = n_states
            
            return self.base_model(optimal_n_states) # return the best model
            
        else: # in the case that all model we fit don't have viable BIC, we return None
            return None
        
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        # get the minimum number of states amaong the samples
        max_n_components = min(self.lengths)
        
        #the self.max_n_component shouldn't be greater than the minimum number of states among the samples
        if max_n_components < self.max_n_components:
            self.max_n_components = max_n_components
            
        # initialize the dictionary that stores DIC value of different n_states choices
        # key: n_states, value: BIC value
        n_states_dict = dict()
        for n_states in range(self.min_n_components, self.max_n_components+1):
            n_states_dict[n_states] = 0
                
        # fit models of different n_states ranging from self.min_n_components to self.max_n_components
        for n_states in range(self.min_n_components, self.max_n_components+1):
            hmm_model = self.base_model(n_states) # fit the model
                
            #update the dictionary
            if hmm_model is not None:
                try:
                    logL = hmm_model.score(self.X, self.lengths)
                    
                    sum_other_logL = 0
                    M = 1
                    for word in self.hwords.keys():
                        if word != self.this_word:
                            X, lengths = self.hwords[word]
                            try:
                                another_logL = hmm_model.score(X, lengths)
                                sum_other_logL += another_logL
                                M += 1
                            except:
                                sum_other_logL += 0
                                M += 0
                    
                    if M > 1:
                        DIC = logL - sum_other_logL / (M - 1)
                    else:
                        DIC = logL
                        
                    n_states_dict[n_states] = DIC
                                
                except: # in the case that the model doesn't have viable logL, we don't update the dictionary
                    n_states_dict[n_states] += 0

        #initialize the optimal_n_state
        optimal_n_states = self.min_n_components
        while optimal_n_states <= self.max_n_components and n_states_dict[optimal_n_states] == 0:
            optimal_n_states += 1   
            
        # loop through the dictionary to find the optimal_n_state that has the lagest DIC
        if optimal_n_states <= self.max_n_components:
            for n_states in n_states_dict.keys():
                if n_states_dict[n_states] != 0 and n_states_dict[n_states] > n_states_dict[optimal_n_states]:
                    optimal_n_states = n_states
            
            return self.base_model(optimal_n_states) # return the best model
            
        else: # in the case that all model we fit don't have viable DIC, we return None
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        # get the minimum number of states amaong the samples
        max_n_components = min(self.lengths)
        
        #the self.max_n_component shouldn't be greater than the minimum number of states among the samples
        if max_n_components < self.max_n_components:
            self.max_n_components = max_n_components
            
        # get the number of samples (sample size) as k
        k = len(self.sequences)
        
        # if there is only one sample, we cannot perform CV, 
        #instead we return the best model that has the largest logL on the single sample
        if k == 1:
            
            # initialize the dictionary that stores logL of different n_states choices
            # key: n_states, value: logL
            n_states_dict = dict()
            for n_states in range(self.min_n_components, self.max_n_components+1):
                n_states_dict[n_states] = 0
                
            # fit models of different n_states ranging from self.min_n_components to self.max_n_components
            for n_states in range(self.min_n_components, self.max_n_components+1):
                hmm_model = self.base_model(n_states) # fit the model
                
                #update the dictionary
                if hmm_model is not None:
                    try:
                        logL = hmm_model.score(self.X, self.lengths)
                        n_states_dict[n_states] += logL
                    except:
                        n_states_dict[n_states] += 0
            
            #initialize the optimal_n_state
            optimal_n_states = self.min_n_components
            while optimal_n_states <= self.max_n_components and n_states_dict[optimal_n_states] == 0:
                optimal_n_states += 1   
            
            # loop through the dictionary to find the optimal_n_state that has the largest logL
            if optimal_n_states <= self.max_n_components:
                for n_states in n_states_dict.keys():
                    if n_states_dict[n_states] != 0 and n_states_dict[n_states] > n_states_dict[optimal_n_states]:
                        optimal_n_states = n_states
            
                return self.base_model(optimal_n_states) # return the best model
            
            else: # in the case that every model we fit either is None or gives None logL, we return None
                return None
                
            
        # initialize the dictionary for storing CV results  
        n_states_dict = dict()
        
        for n_states in range(self.min_n_components, self.max_n_components+1):
            #key: n_states, value: [sum of valid CV logLs, number of valid CV results]
            n_states_dict[n_states] = [0,0] 
        
        # if the sample size is 2, use 2-fold CV
        # if the sample size is greater than 2, then sue 3-folod CV
        if k == 2:
            split_method = KFold(n_splits = k)
        else:
            split_method = KFold()
            
        #Perform Cross-Validation
        for train_idxes, test_idxes in split_method.split(self.sequences):
            
            #split train and test sets for each iteration of CV
            train_X = np.array(self.sequences[train_idxes[0]])
            train_lengths = [self.lengths[train_idxes[0]]]
            test_X = np.array(self.sequences[test_idxes[0]])
            test_lengths = [self.lengths[test_idxes[0]]]

            for idx in train_idxes[1:]:
                train_X = np.concatenate((train_X, self.sequences[idx]))
                train_lengths.append(self.lengths[idx])
    
            for idx in test_idxes[1:]:
                test_X = np.concatenate((test_X, self.sequences[idx]))
                test_lengths.append(self.lengths[idx])
            
            # fit models of different choices of n_states
            for num_states in range(self.min_n_components, self.max_n_components+1):
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                except:
                    hmm_model = None
                
                # update the dictionary
                if hmm_model is not None:
                    try:
                        logL = hmm_model.score(test_X, test_lengths)
                        n_states_dict[num_states][0] += logL
                        n_states_dict[num_states][1] += 1
                    except: #in the case that the model doesn't generate logL, the dictionary doesn't update in this round
                        n_states_dict[num_states][0] += 0
                        n_states_dict[num_states][1] += 0
        
        # averaging the CV logL, update the dictionary
        for n_states in n_states_dict.keys():
            if n_states_dict[n_states][1] > 0:
                n_states_dict[n_states][0] = n_states_dict[n_states][0] / n_states_dict[n_states][1]
        
        # loop through the dictionary to find the optimal_n_states
        optimal_n_states = self.min_n_components
        while optimal_n_states <= self.max_n_components and n_states_dict[optimal_n_states][0] == 0:
            optimal_n_states += 1
        
        if optimal_n_states <= self.max_n_components:
            for n_states in n_states_dict.keys():
                if n_states_dict[n_states][0] != 0 and n_states_dict[n_states][0] > n_states_dict[optimal_n_states][0]:
                    optimal_n_states = n_states
            
            return self.base_model(optimal_n_states)
        
        # In the case when every model we fit either is None or gives None CV result of logL,
        # we will compute the best model as we did in the case of k = 1
        else:
            n_states_dict = dict()
            for n_states in range(self.min_n_components, self.max_n_components+1):
                n_states_dict[n_states] = 0
            for n_states in range(self.min_n_components, self.max_n_components+1):
                hmm_model = self.base_model(n_states)
                
                if hmm_model is not None:
                    try:
                        logL = hmm_model.score(self.X, self.lengths)
                        n_states_dict[n_states] += logL
                    except:
                        n_states_dict[n_states] += 0
            
            optimal_n_states = self.min_n_components
            while optimal_n_states <= self.max_n_components and n_states_dict[optimal_n_states] == 0:
                optimal_n_states += 1   
                
            if optimal_n_states <= self.max_n_components:
                for n_states in n_states_dict.keys():
                    if n_states_dict[n_states] != 0 and n_states_dict[n_states] > n_states_dict[optimal_n_states]:
                        optimal_n_states = n_states
            
                return self.base_model(optimal_n_states)
            
            else:
                return None
    
            
            
            
