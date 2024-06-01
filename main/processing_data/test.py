import sys
sys.path.insert(1, 'D:/Workspace/Project_VNNIC')
from main.featureURL.lexical_feature import *
from getLexicalFeature import * 
url ="beet8.com"
print(getLexicalInputNN(url))