import  numpy as np
import scipy
from scipy import special
from scipy.special import expit as logist
import sys
import cPickle as pickle

# This is the code for the reversal learning task.
# The reversal learning task is essentially equivalent to the one-shot learning
# task, except that halfway through the episode, we revert the labels of the
# two patterns, show each pattern with updated label once, and go on.


# Note that we include code for gradient checking (try %run oneshot.py GRADIENTCHECKING 1)


np.set_printoptions(suppress=True, precision=8)  # Because engineering notation gets old fast


xsize=10    # Input layer: 8 pattern bits + 2 label bits.
zsize=2     # Output layer
# ysize (the size of the hidden layer) is defined in g, and thus changeable from command line. 


# All the following options are modifiable from the command line
g = {
'GRADIENTCHECKING': 0,
'NBITER': 20,       # Number of timestesp per episode      
'YSIZE': 2,         # Number of neurons in the hidden (y) layer 
'LEARNPERIOD': 2,   # Learning period during which the desired response is shown at the same time as the patterns. Set to 2 for one-shot learning.
'POSWEIGHTS': 0,    # If you want to enforce all weights to be positive (don't!)
'UPDATETYPE': 'RMSPROP',
'ALPHATRACE' : .98, # The Gamma parameter in the paper (time constant for the exponential decay of the running average of input-output products fot the Hebbian traces)
'NBEPISODES': 10000,  # Total number of episodes. For gradient checking, always 3 (see below). 
'RNGSEED' : 0,      # Random seed for reproducibility
'ETA' : .01,       # Learning rate
'MAXDW' : .01,      # Maximum absolute value of parameter modification after every episode (see below)
'WPEN' : 1e-4,      # Coefficient of the L1 penalty on weights for regularization.
'LINEARY': 0,       # If you want linear rather than tanh output on the hidden layer, for debugging or gradient-checking
'TEST': 0           # If you want to run using weights from a saved file (see below)
}


# Parsing command line arguments
argpairs = [sys.argv[i:i+2] for i in range(1, len(sys.argv), 2)]
for argpair in argpairs:
    if not (argpair[0] in g):
        raise ValueError("Error, tried to pass value of non-existent parameter "+argpair[0])
    if (argpair[0] == 'UPDATETYPE') or (argpair[0] == 'STIMCORR') :  # String (non-numeric) parameters
        g[argpair[0]] = argpair[1]
    else:
        g[argpair[0]] = float(argpair[1])

# Initialization!
ysize = int(g['YSIZE'])
np.random.seed(int(g['RNGSEED']))
wxy = np.random.randn(ysize, xsize) * .1
by = np.random.randn(ysize) * .1
wyz =  np.abs(np.random.randn(zsize, ysize) * .1)
bz = np.random.randn(zsize) * .1
alpha = np.random.randn(ysize, xsize) * .1
hebb = np.zeros((ysize, xsize)) 
mwxy, mwyz, malpha , mbz, mby = np.zeros_like(wxy), np.zeros_like(wyz), np.zeros_like(alpha), np.zeros_like(bz), np.zeros_like(by)



if g['GRADIENTCHECKING']:
    g['NBITER'] = 20
    g['LEARNPERIOD'] = 10
    g['NBEPISODES'] = 3
    FROZENINPUTSIZE = 500; frozeninputs = .2 * np.random.randn(xsize, FROZENINPUTSIZE)  # For debugging / gradient checking
    WINC = 0.003 *  np.random.randn(ysize, xsize)
    ALPHAINC = 0.003 * np.random.randn(ysize, xsize)
    WINC.fill(.005); ALPHAINC.fill(.005)
    #ALPHAINC.fill(0)
    #WINC.fill(0)


# DEBUGGING !
#alpha.fill(0)

archy = []
archz = []
errs=[]
archerrs = []

if g['TEST']:
    testdir='./'
    with open(testdir+'data.pkl', 'r') as handle:
              (wxy, wyz, alpha, by, bz, hebb, errs, g) = pickle.load(handle)
    g['TEST'] = 1  # because the data file will obliterate g



for episode in range(g['NBEPISODES']):
 
    # The system does work with 0/1 patterns, but less well / need a somewhat longer learning period.
    # -1/1 patterns:
    pattern1 = 2 * (np.random.rand(xsize-2, 1) > .5).astype(int) -1
    # A pattern of 0s and 1s with at least two 1s:
    #while True:
    #    pattern1 = (np.random.rand(xsize-2, 1) > .5).astype(int)
    #    if np.sum(pattern1) > 1:
    #        break
    #pattern2 = pattern1.copy()
    #flipbit = np.random.randint(pattern2.size)
    ##pattern2[flipbit] =  1 - pattern2[flipbit]   # For 0/1
    #pattern2[flipbit] =   - pattern2[flipbit]   # for -1/1

    # A random pattern that differs from pattern 1 in at least one place
    pattern2 = 2 * (np.random.rand(xsize-2, 1) > .5).astype(int) -1
    while not np.any(pattern1 != pattern2):
        pattern2 = 2 * (np.random.rand(xsize-2, 1) > .5).astype(int) -1
    
    if g['GRADIENTCHECKING']:
        wxy += WINC
        alpha += ALPHAINC
        pattern1.fill(1)
        pattern2.fill(1)
        pattern2[1] =-1 
        np.random.seed(int(g['RNGSEED']))    # We need to use the same random seed for all 3 episodes!
    
    dydas=[]
    dydws=[]
    xs , ys, zs, tgts = [], [], [], [] 
    hebb.fill(0)

    # Run the episode!
    for n in range(int(g['NBITER'])):

        # DEBUGGING
        #alpha.fill(0)
        
        # Reversal !
        if n == g['NBITER'] / 2:
            pattern3 = pattern2.copy()
            pattern2 = pattern1.copy()
            pattern1 = pattern3.copy()


        x = np.zeros((xsize, 1)).astype(int)
        if n%2 == 0:     
            pattern = pattern1
        else:
            pattern = pattern2

        x[2:] = pattern[:]
        
        # The target output is the label for this pattern - 01 or 10
        tgt = np.zeros((2, 1))
        tgt[n % 2] = 1

        # We only show the labels during the learning periods, both at the start and after reversal
        if n < g['LEARNPERIOD'] or (n >= g['NBITER'] / 2 and n < g['NBITER'] / 2 + g['LEARNPERIOD']):
            x[:2] = tgt[:]

        
        # DEBUGGING ! Also for gradient checking...
        #x = np.zeros((xsize, 1)).astype(int)
        #x = np.ones((xsize, 1)).astype(int)
        #x = np.random.randn(xsize, 1)
        #x[n%2] = 1

        #x = frozeninputs[:, n % FROZENINPUTSIZE, None]  # For debugging / gradient-checking
        #y = np.dot(wxy, x) + np.dot (alpha*hebb, x)  # Linear output, for debugging 
        #z = np.dot(wyz, y)
        


        if g['LINEARY']:
            y = np.dot(wxy, x) + np.dot (alpha*hebb, x) + by[:, None]  # Linear output, for DEBUGGING !
        else:
            y = np.tanh(np.dot(wxy, x) + np.dot (alpha*hebb, x) + by[:, None])   # Tanh nonlinearity on the y
            #y = logist( np.dot(wxy, x) + np.dot (alpha*hebb, x) + by[:, None] )   # Logistic nonlinearity on the y
        #z = np.tanh(np.dot(wyz, y) + bz)   # Tanh nonlinearity on the z
        #z = 1.0 / (1.0 + np.exp(-(np.dot(wyz, y) + bz)))  # Logistic nonlinearity on the z 
        zraw = np.dot(wyz, y) + bz[:, None]  
        z = np.exp(zraw) / (1e-8 + np.sum(np.exp(zraw)))  # Softmax competition between the z


        # Okay, now we comnpute the quantities necessary for the actual BOHP algorithm.

        # Pre-compute the gammas for the gradients of the Hebbian traces (see paper). Note that n is the number of steps already done.
        gammas = [(1.0 - g['ALPHATRACE'])*(g['ALPHATRACE']**(n-(xx+1))) for xx in range(n)]  

        # Gradients of y(t) wrt alphas (dyda):
        summand=0
        for k in range(xsize):
            summand += alpha[None, :, k].T * x[k] * sum([prevx[k] * prevdy *gm for prevx, prevdy, gm in zip(xs, dydas, gammas)]) # Gradients of the Hebbian traces wrt the alphas
        dyda = summand + x.T * hebb   # Note that dyda is a matrix, of the same shape as wxy (there is one gradient per alpha, and one alpha per connection)
        if not g['LINEARY']:
            dyda = (1 - np.tanh(y) * np.tanh(y)) * dyda  
            #dyda = (1 - logist(y)) * logist(y) * dyda  

        # Gradients of y(t) wrt ws (dydw):
        summand=0
        for k in range(xsize):
            summand += alpha[None, :, k].T * x[k] * sum([prevx[k] * prevdy *gm for prevx, prevdy, gm in zip(xs, dydws, gammas)]) 
        dydw = summand + x.T * np.ones_like(wxy)
        if not g['LINEARY']:
            dydw = (1 - np.tanh(y) * np.tanh(y)) * dydw  
            #dydw = (1 - logist(y)) * logist(y) * dydw  

        # Store relevant variables of this timestep (x, y, z, gradients)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        tgts.append(tgt)
        dydas.append(dyda)
        dydws.append(dydw)

        # Update the Hebbian traces (remember that 'hebb' is a matrix of same dimensionality as wxyi - one hebbian trace per connection)
        #hebb += (1.0 - g['ALPHATRACE']) * (np.dot(np.arctanh(y), x.T) - hebb)
        hebb += (1.0 - g['ALPHATRACE']) * (np.dot(y, x.T) - hebb)
        # End of the episode!

    #raise(ValueError)

    # Archive the ys of this episode for debugging
    ys = np.hstack(ys)
    zs = np.hstack(zs)
    xs = np.hstack(xs)
    tgts = np.hstack(tgts)
    dydas = np.dstack(dydas)
    dydws = np.dstack(dydws)
    
    #print hebb

    archy.append(ys)
    archz.append(zs)
    
    # Now that the episode is completed, we can run some backpropagation. 

    # Loss: -log(p(correct)), where p(x) is the proba computed for option x by the softmax (i.e. the z)
    errors = -np.log(np.choose(np.argmax(tgts, axis=0), zs))
    errors[:g['LEARNPERIOD']].fill(0)       # We don't care about early episode, exploratory "learning" period.....  Not strictly needed, but makes things a bit better.
    errors[g['NBITER']/2:g['NBITER']/2+g['LEARNPERIOD']].fill(0)
    archerrs.append(errors)
    
    # Now we compute the gradient of the error wrt z at each timestep:
    dedzsraw = zs.copy() - tgts.copy()             # Derivative of loss (-log(p(correct))) through softmax: the computed probabilities minus the target (0 or 1) probabilities (see http://cs231n.github.io/neural-networks-case-study/#grad )
    dedzsraw[:, :g['LEARNPERIOD']].fill(0)
    dedzsraw[:, g['NBITER']/2:g['NBITER']/2+g['LEARNPERIOD']].fill(0)
    #dedzsraw = dedzs * zsflat * (1 - zsflat)    # Gradient through logistic nonlinearity
    #dedzsraw = dedzs * (1 - zsflat * zsflat)   # Gradient through tanh nonlinearity
    #dedzsraw = dedzs.copy() # Gradient through linear output, for debugging
    #dedzsraw = zsflat - tgts   # Experimental hack attempt at cross-entropy-ish error. Didn't seem to make much difference
    

    # We backpropagate this through the yz weights to get the gradient of the error wrt the y's at each timestep:
    dedys =  np.dot(wyz.T, dedzsraw)    # Should have dimensions ysize, nbtimesteps

    dedysrawthroughtanh = dedys * (1 -  ys * ys)    # Gradient through tanh nonlinearity. This is needed for the biases.
    dedysraw = dedys.copy()                         # Since the gradients dyda and dydw already includes the tanh nonlinearity, we don't need it here.

    if g['GRADIENTCHECKING']:
        if episode == 1:   
            rdydas = dydas.copy()
            rdydws = dydws.copy()
            rdedys = dedysraw.copy()
            rdedas = dedysraw[:, None, :] * rdydas
            rdedws = dedysraw[:, None, :] * rdydws
    
    # Now we use the computed gradients for actual weight modification, using the quantities computed at each timestep during the episode!
    if (not g['TEST']) and (not g['GRADIENTCHECKING']) and episode < g['NBEPISODES']-500:
        # First, the biases by and bz:
        dbz = np.sum(dedzsraw, axis=1)    
        dby = np.sum(dedysrawthroughtanh, axis=1)
        #dby = np.sum(dedysraw, axis=1)
        # Then the fixed weight matrices wxy and wyz, and the alpha's:
        #dwyz = np.dot(ysflat, dedzsraw[:, None]).T
        dwyz = dedzsraw.dot(ys.T)  #.... ?

        dalpha = np.sum(dedysraw[:, None, :] * np.array(dydas), axis=2)
        dwxy =  np.sum(dedysraw[:, None, :] * np.array(dydws), axis=2)
        
        for param, dparam, mem in zip([wxy, wyz, alpha, by, bz], 
                                    [dwxy, dwyz, dalpha, dby, dbz],
                                    [mwxy, mwyz, malpha, mby, mbz] ):
            #mem = .99 * mem + .01 * dparam * dparam   # Does NOT update the m-variables - head.desk()
            if episode == 0:
                mem +=  dparam * dparam 
            else:
                mem +=  .01 * (dparam * dparam - mem)    # Does update the m-variables
            RMSdelta = -g['ETA'] * dparam / np.sqrt(mem + 1e-8) 
            #print np.mean( np.abs(RMSdelta) >  g['MAXDW'])
            np.clip(RMSdelta, -g['MAXDW'], g['MAXDW'], out = RMSdelta)
            
            if g['UPDATETYPE'] == 'RMSPROP':   # RMSprop update
                delta = dparam / np.sqrt(mem + 1e-8) 
            elif g['UPDATETYPE'] == 'SIMPLE':  # Simple SGD
                delta = dparam
            else:
                raise ValueError('Wrong / absent update type!')
            
            # Notice the clipping
            param += np.clip( -g['ETA'] * delta , -g['MAXDW'], g['MAXDW'])   
            #print np.mean(np.abs(g['ETA'] * delta) > g['MAXDW'])


        #  L1-penalty on all weights and alpha, for regularization (really useful!)
        alpha -= g['WPEN'] * np.sign(alpha)
        wxy -= g['WPEN'] * np.sign(wxy)
        wyz -= g['WPEN'] * np.sign(wyz)

        if g['POSWEIGHTS']:   # Just don't!
            wxy[wxy<1e-6] = 1e-6
            wyz[wyz<1e-6] = 1e-6
            alpha[alpha<1e-6] = 1e-6

    # Log the total error for this episode
    meanerror = np.mean(np.abs(errors[g['LEARNPERIOD']:]))

    # Every 10th episode, display a message and update the output file
    if episode % 10 == 0:
        print "Session #", episode, " - mean abs. error per timestep (excluding learning period): ", meanerror
        print "Errors at each timestep (should be 0.0 for early learning period):"
        print errors
        np.savetxt("errs.txt", np.array(errs))
        with open('data.pkl', 'wb') as handle:
              pickle.dump((wxy, wyz, alpha, by, bz, hebb, errs, g), handle)
        ## For reading:
        #with open('data.pkl', 'r') as handle:
        #      (wxy, wyz, alpha, by, bz, hebb, errs, g) = pickle.load(handle)
    errs.append(meanerror)

    # End loop over episodes


# This checks the gradient on ys 
if g['GRADIENTCHECKING']: 
    print "Predicting change in y:"
    #diff = np.arctanh(archy[2]) - np.arctanh(archy[0]) 
    diff = archy[2] - archy[0]
    #diff = np.array(archy[1]) - np.array(archy[0]) 
    #calcdiffa = (1 - np.tanh(archy[1]) * np.tanh(archy[1])) * np.sum(rdydas*ALPHAINC[:, :, None]*2, axis=1)
    #calcdiffw = (1 - np.tanh(archy[1]) * np.tanh(archy[1])) * np.sum(rdydws*WINC[:, :, None]*2, axis=1)
    calcdiffa =  np.sum(rdydas*ALPHAINC[:, :, None]*2, axis=1)
    calcdiffw =  np.sum(rdydws*WINC[:, :, None]*2, axis=1)
    calcdifftot = calcdiffa + calcdiffw
    print " Measured diff.:"
    print diff
    print "Calculated diff.:"
    print calcdifftot
    print "Relative Absolute Error:"
    zd = np.abs(diff - calcdifftot) / (diff + 1e-8)
    print zd
    print "(Mean/std: ", np.mean(zd), np.std(zd)

    print "Predicting change in final error (-log(p(correct)) in softmax/z layer):"
    differr = archerrs[2] - archerrs[0]  
    diffz = archz[2] - archz[0]  
    diffy = archy[2] - archy[0] 
    predicteddifferrfromdiffy = np.sum(rdedys * diffy, axis=0)    
    print " Measured diff. err:"
    print differr.T
    print "Calculated diff. from the diffy:"
    print predicteddifferrfromdiffy
    #dedas = dedysraw.T[:, :, None] * rdydas
    #dedws = dedysraw.T[:, :, None] * rdydws
    calcdifferr_a = np.sum(np.sum(rdedas*ALPHAINC[:, :, None]*2, axis=1), axis=0)
    calcdifferr_w = np.sum(np.sum(rdedws*WINC[:, :, None]*2, axis=1), axis=0)
    calcdifferr_tot = calcdifferr_a + calcdifferr_w
    print "Calculated diff from inputs:"
    print calcdifferr_tot

