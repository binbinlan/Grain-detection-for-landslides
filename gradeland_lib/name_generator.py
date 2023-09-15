import random as rnd
import cv2
import numpy as np

def makeletterlists():
    vowels = 'aeiou'
    alphabet= 'abcdefghilmnopqrstuvz'
    vowlist = []
    consonlist = []
    for i in alphabet:
        if i in vowels:
            vowlist.append(i)
        else:
            consonlist.append(i)
    return vowlist,consonlist

def makeaname(wordlen,vowels, consonnants,vocality=0.5):
    newname = ''
    while len(newname)<wordlen:
        coint = rnd.random()
        if len(newname)>=2:
            if newname[-1] in vowels and newname[-2] in vowels:
                newname+=consonnants[rnd.randint(0,len(consonnants)-1)]
            if newname[-1] in consonnants and newname[-2] in consonnants:
                newname+=vowels[rnd.randint(0,len(vowels)-1)]
        if coint>vocality:
            newname+=consonnants[rnd.randint(0,len(consonnants)-1)]
        else:
            newname+=vowels[rnd.randint(0,len(vowels)-1)]
    if len(newname)>wordlen:
        newname = newname[:-1]
    return newname

def makeacolor(brightness = 128):
    newcolor = []
    newcolor.append(rnd.randint(0,255))
    newcolor.append(rnd.randint(0,(brightness*3)-newcolor[0]))
    newcolor.append((brightness*3)-newcolor[0]-newcolor[1])
    return newcolor

if __name__== '__main__':
    vowels, consonnants = makeletterlists()
    vocality = 0.5
    wordlen = 4
    for i in range(15):
        print(makeaname(rnd.randint(3,10)))

    definition = 800
    blackcanvas = np.zeros([800,800,3],dtype= np.uint8)
    for i in range(100):
        newcolor = makeacolor()
        cv2.circle(blackcanvas,[rnd.randint(0,definition),rnd.randint(0,definition)],radius=8,color=newcolor,thickness=6)
    cv2.imshow('coriandoli',blackcanvas)
    cv2.waitKey(0)
    