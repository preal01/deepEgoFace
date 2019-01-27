#copyrights in tsne.py
from . import tsne

import os
import errno

import re
import random

import numpy as np
from sklearn import model_selection
import pylab





class logger:
    """
    Class made to record msgArgs on add in (str)filePath
    """
    def __init__( self, filePath):
        dirName = os.path.dirname(filePath)
        try: os.makedirs(dirName)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dirName): pass
            else: raise
        self.file = open( filePath, "a+")

    def __del__(self):
        self.file.close()

    def add( self, *msgArgs):
        for msgArg in list(msgArgs):
            self.file.write(str(msgArg)+",")
        self.file.write("\n")


def mergeCorpusAndOtherEmbbedingsClasses( ecCorpusFile, labels, eOtherFile, ecALLFile):
    """
    Take embeddings and class from corpus in (str)ecCorpusFile
    and merge with the embeddings in (str)eOtherFile
    into (str)ecALLFile
    (str)labels used to assign correct class
    """
    oi = labels.index("other")
    resFile = open(ecALLFile,"w+")
    with open(ecCorpusFile) as f:
        for i, l in enumerate(f):
            resFile.write(l)
    nbEmbeddings = i + 1
    nbPerClass = int(nbEmbeddings/(len(labels)-1))

    eOther = []
    with open(eOtherFile) as f:
        for i, l in enumerate(f):
            eOther.append(re.sub( '\n', '', l)+str(oi)+"\n")

    nbDel = len(eOther) - nbPerClass
    for i in range(0,nbDel):
        index = random.randrange(len(eOther))
        del eOther[index]

    for l in eOther:
        resFile.write(l)
    resFile.close()





def getLabels(labeledFaceFolder):
    """
    return an array containing classes name based on a hand organized corpus
    where each class as its own folder
    """
    dirs = os.listdir(labeledFaceFolder)
    labels = []
    for dname in dirs:
        labels.append(dname)
    labels.append("other")
    return labels



def linkEmbeddingsAndClass( labeledFaceFolder, embeddingsFile, resFile):
    """
    associate embeddings contained in (str)embeddingsFile
    and the images contained in a hand organized corpus
    into (str)resFile
    """
    classes = []
    dirs = os.listdir(labeledFaceFolder)
    cls = 0
    for dname in dirs:
        files = os.listdir(labeledFaceFolder+"/"+dname+"/")
        for fname in files:
            classes.append([ fname, cls])
        cls = cls + 1

    file  = open(embeddingsFile, "r")
    resFile = open(resFile,"w+")
    for line in file:
        elems = line.split(",")
        try:
            index = [el[0] for el in classes].index(elems[0]+"-"+elems[1]+"-"+elems[2]+".png")
        except:
            pass
        else:
            resFile.write( re.sub( '\n', '', line)+str(classes[index][1])+"\n")
    file.close()
    resFile.close()




def visualizeEmbeddings(embbedingsClassesPath,size=1):
    """
    use tsne for visualising a dimensionally reduced set of embeddings
    """
    file = open(embbedingsClassesPath,"r")
    data = []
    for line in file:
        data.append(line)
    file.close()
    x = []
    y = []
    for elems in data:
        elemsL = elems.split(",")
        x.append([float(el) for el in elemsL[3].split(" ")])
        y.append(int(elemsL[4]))
    xt, x, yt, y = model_selection.train_test_split(x, y, test_size=size, random_state=7)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    Y = tsne.tsne(x, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, y)
    pylab.savefig('embeddings-reduction-'+str(int(100*size))+'p.png')
    pylab.show()
