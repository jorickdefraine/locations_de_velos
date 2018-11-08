'''
si vous avez bien pull, ce message s'affiche.
'''

from processing import OpenData, affiche_corr, affiche_hm, feature_imp, firstLine

print(OpenData())
#affiche_corr(OpenData(), 16)
affiche_hm(OpenData())
print("Première ligne")


#print(firstLine(0))
#feature_imp(OpenData(), firstLine(0))