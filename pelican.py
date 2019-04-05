# -*- coding: utf-8 -*-
#!/usr/bin/env python

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

import getopt
import argparse
import time

import sys, getopt

import h5py, numpy as np, os

import itertools
# import matplotlib.pyplot as plt
import eventHelper as eH
# import eventHelper as ed
from eventHelper import f_events
from MLpredictor import MLpred
from quotes import quotes
from progress import progress
from keras.preprocessing.sequence import pad_sequences
from VCNN1 import predictLabel
from DPredict import DP
from numpy import zeros, newaxis
from Fast5_parsing import fast5_parse
from file_indexing import file_index
from keras import backend as K
# import psutil


starttt = time.time()

print("For CNN based prediction Tensorflow will be used.."+'\n')
# ########################################### file control ########################################

parser = argparse.ArgumentParser(description='PELICAN--> Fast5 Analysis for modified A Nucleotides')


parser.add_argument('-i', action='store',
                    dest='path_input',
                    help='Provide Fast5 folder path --> ex: /path/')

parser.add_argument('-n', action='store',
                    dest='nucleotide_Base',
                    help='Provide nucleotide base type to predict modifications(ex:A)--> Default: A (Do Not use this option in current version)')

parser.add_argument('-b', action='store',
                    default=3,
                    dest='file_batch',
                    help='Process files in batches depending on your available resources..smaller batch size is recommended for less computational resources')

parser.add_argument('-T', action='store_false',
                    default=False,
                    dest='WriteToFast5',
                    help='Default: Do not writes modified sequences to fast5 files under mFastq --> Set -f to turn it off')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

results = parser.parse_args()
inp = results.path_input
nb= results.nucleotide_Base
overw=results.WriteToFast5

fBatch_size=results.file_batch
fBatch_size=int(fBatch_size)


print('\n'+"### "+quotes()+" ###"+'\n')
##############################################################

l_filename=file_index(inp)

# for filenames in os.listdir(str(inp)):
#     if filenames.endswith(".fast5"):
#         l_filename.append(inp+filenames.replace('\n',''))
print("File Processing Started..."+'\n')
print('Total Number of files to be processed: '+str(len(l_filename)))


############ Rock n Roll --> show begins Fast5 file processing ####################################    

 
cnt=0
ef=str(len(l_filename))

# def get_mem_usage():                                                                                                                               
#     process = psutil.Process(os.getpid())                                                                                                          
#     return process.memory_info()                                                                                                                   



fCounter=0
while fCounter < len(l_filename):
    start = fCounter
    end = fCounter+fBatch_size
    fBatch = l_filename[start:end]

    holder=[]
    sigholder=[]
    for r in fBatch:
        try:
            events, fastq_decoded, raw_signal = fast5_parse(r)
        except AttributeError:
            continue 
        final_eves=eH.event_scrapper(events)    ### event scrapping
        # print(final_eves)


        
        seq_no=3    ### inititalize sequence number

        for e, i in enumerate(final_eves):

            seq_no=seq_no+int(i[1]) ### update sequence number with steps
            # print(len(''.join(fastq_decoded[1])))
            len_seq=len(''.join(fastq_decoded[1]))  ### length of sequence
        ##### Tail pass
            if e > 2 and e < len(final_eves)-2 and i[0][2:][:1] == 'A':
                f_e=f_events(final_eves,e)
                if len(f_e) != 0:
                    start=[]
                    end=[]
                    for s in f_e:
                        for t in range(5):
                            start.append(s[t][2])
                            end.append(s[t][4])
                    #         print(s[t][2])
                    # print(min(start))
                    # print((max(end)))
                    # print(raw_signal[min(start):][:(max(end)-min(start))+1])            
                    min_st=min(start)
                    max_stl=(max(end)-min(start))+1

                    sig='_'.join(map(str, raw_signal[min_st:][:max_stl]))
                    a_seq_no=(len_seq-seq_no)+1   ### transverse seqno

                    if len(sig) <= 1000:
                        # print(fastq_decoded[0])
                        id=fastq_decoded[0]

                        # print(str(len_seq)+' '+str(a_seq_no)+' '+str(seq_no)+' '+str(min_st)+'_'+str(max_stl)+' '+str(len(sig)))
                        # print()
                        holder.append(id+' '+str(seq_no)+' '+str(a_seq_no))
                        sigholder.append(sig.split('_'))
        cnt=cnt+1
    fsignals=[list(map(int, i)) for i in sigholder]
    psignals=pad_sequences(fsignals, padding="post", maxlen=1000)
    # print(X.shape)
    X = psignals[:, :, newaxis]
    # print(X.shape)
    
    plabels, proba = DP(X,len(X))
    # print(proba)
    filename = 'm1A_Prediction_output3.txt'

    if os.path.exists(filename):
        a_write = 'a' # append if already exists
    else:
        a_write = 'w' # make a new file if not
    file3=open(filename,a_write)
    for s, v, m in zip(holder,plabels, proba):
        if m[1] > 0.5:
            # print(s,v)
            sv=s+' '+str(v)+' '+str(m[1])
            # print(sv)
            file3.write(sv+'\n')
    file3.close()
    progress(cnt, ef, status='Processing..Please wait..')

    fCounter += fBatch_size

    # time.sleep(0.5)
print("\nTotal files processed: ", cnt)
# print((time.time() - start))
stopttt = time.time()
print(stopttt-starttt)

# mem = get_mem_usage() 
# print('mem: {}'.format(t, mem))
