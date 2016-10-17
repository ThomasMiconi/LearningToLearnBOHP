# Submit jobs to the cluster. 



import sys
import os
import shutil

allopts = [


       # With RMSprop, you want MAXDW to be ~ 3 times ETA or more (5 times means clipping will be extremely rare after a while)
       # Without RMSprop (simple SGD), the steps have different sizes for wxy and alpha.. With a same ETA, easier to hit MAXDW with wxy than with alpha...
       # Use RMSprop throughout!


        # For the two-layer, uncorelated-stimuli case


        "ALPHATRACE .98",
        "ALPHATRACE .95",
        "ALPHATRACE .75"
        
       
        ]


for optionz in allopts:

    #dirname = "trial-ref-" + optionz.replace(' ', '-')
    #dirname = "trial-fixedsize-CMN-" + optionz.replace(' ', '-')
    dirname = "trial-logistic-withz-" + optionz.replace(' ', '-')

    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    print os.getcwd()

    for v in range(20):
        os.mkdir("v"+str(v))
        os.chdir("v"+str(v))
        CMD = "bsub -q short -W 8:00 -eo e.txt -g /rnn /opt/python-2.7.6/bin/python ../../reversal.py " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 4:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../rnn.py " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 6:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../min-char-rnn-param.py " + optionz + " RNGSEED " + str(v) # For fixed-size
        #print CMD
        retval = os.system(CMD)
        print retval
        os.chdir('..') 
    
    os.chdir('..') 


    #print dirname
    #for RNGSEED in range(2):
    #st = "python rnn.py COEFFMULTIPNORM " + str(CMN) + " DELETIONTHRESHOLD " + str(DT) + " MINMULTIP " \
    #+ str(MMmultiplierofDT*DT) + " PROBADEL " + str(PD) + " PROBAADD " + str(PAmultiplierofPD * PD) \
    #+ " RNGSEED " + str(RNGSEED) + " NUMBERMARGIN " + str(NM)




