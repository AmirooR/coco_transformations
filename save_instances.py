# coding: utf-8
from pycocotools.coco import COCO
import numpy as np
import cPickle as pickle
import random

dataDir='..'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

saveDir='data_%s' % dataType
saveName='%s/instances.pkl' % saveDir

coco=COCO(annFile)
annIds = coco.getAnnIds(iscrowd=False)
anns = coco.loadAnns(annIds)
print len(anns)
random.shuffle(anns)
with open(saveName, 'wb') as output:
	pickle.dump(anns, output, -1)
