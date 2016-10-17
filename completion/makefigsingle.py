import glob
import numpy as np
import matplotlib.pyplot as plt

#font = {#'family' : 'normal',
##                'weight' : 'bold',
#                        'size'   : 10}
#plt.rc('font', **font)

plt.ion()
np.set_printoptions(precision=3, suppress=True)


dirz = glob.glob('trial-*TRACE-.75')
#dirz = glob.glob('trial-logistic-withz-ETA-0.003-MAXDW-0.01-WPEN-1e-4-UPDATETYPE-RMSPROP-POSWEIGHTS-0-STIMCORR-EXCLUSIVE-YSIZE-1-LEARNPERIOD-0-ALPHATRACE-.95')
dirz.sort()
NBPLOTS = len(dirz)
SS = np.ceil(np.sqrt(NBPLOTS))

plt.figure(figsize=(4, 3), dpi=100, facecolor='w', edgecolor='k')

#plt.figure()

#trial-withz-ETA-0.01-MAXDW-0.03-WPEN-3e-4-UPDATETYPE-RMSPROP-POSWEIGHTS-0-STIMCORR-UNCORRELATED-YSIZE-2-LEARNPERIOD-0-ALPHATRACE-.95
nplot = 1
perfs = []
nbneurs = []
dirs = []
colorz=['b', 'b', 'b', 'r', 'g']
for (num, droot) in enumerate(dirz):
    t = []
    for v in range(20):
        dfull = droot + "/v" + str(v)
        try:
            z = np.loadtxt(dfull+"/errs.txt")
        except IOError:
            print "error loading "+dfull+"/errs.txt"
            continue
        #z=z[::10]   # For clarity, we only show every 10th
        #z=z[:800, :]
        #if len(z) == 251:
        #    t.append(z)
        #else:
        #    print len(z)
        t.append(z)
    t = np.vstack(t)
    tmean = np.mean(t, axis=0)
    tstd = np.std(t, axis=0)
    tmedian = np.median(t, axis=0)
    tq25 = np.percentile(t, 25, axis=0)
    tq75 = np.percentile(t, 75, axis=0)
    
    ax = plt.subplot(SS, SS, nplot)
    #ax.set_title(num)
    #plt.fill_between(range(len(tmean)), tq25, tq75, linewidth=0.0, color='blue', alpha=0.5)
    #xx = 10*np.arange(tmean.size)
    xx = np.arange(tmean.size)
    plt.fill_between(xx, tq25, tq75, color='gray') #, alpha=0.3)
    plt.plot(xx, tmedian, color='black')
    #plt.axis([0, tmean.size*10, 0, 1.0])
    #plt.axis([0, tmean.size, 0, 1.0])
    plt.xlabel('Episode #')
    plt.ylabel('Error')

    p1 = int(tmean.size / 3)
    p2 = 2*int(tmean.size / 3)
    p3 = -1

    print num, p1, ':', tmean[p1], p2, ':', tmean[p2], p3, ':', tmean[p3], droot
    perfs.append([tmean[p1], tmean[p2], tmean[p3]])
    nbneurs.append([tmean[p1], tmean[p2], tmean[p3]])
    dirs.append(droot)

    nplot += 1

print "Data read."

perfs = np.array(perfs)
p = perfs[:,1]
nbneurs = np.array(nbneurs)
dneur = nbneurs[:, 1] - nbneurs[:,2]
ord = np.argsort(p)
data = np.vstack((ord, dneur[ord], p[ord])).T

plt.show()
plt.savefig('fig1.png', bbox_inches='tight')
