# %% [markdown]
# Ce programme de Machine Learning a pour but de déterminer les fraudes bancaire à partir du jeux de donné "creditcard"

# %%
import pandas as pd 
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt 
feature = pd.read_csv('creditcard.csv' ,  sep=",")
print(feature)


# %% [markdown]
# Nettoyage & Exploration des donnés

# %%
print(feature.info()) #Vérifie qu'il n'ya pas de valeurs manquantes 
print(feature.shape) #Nous donnes la taille de la base de donnée
print(feature.describe()) #Donne les caractéristiques statistiques.




# %% [markdown]
# Visualisations & Corrélations des données

# %%
feature_fraudes = feature[feature.Class== 1] # prend  dans la colonne class  les fraudes (celle ou class==1)
plt.figure(figsize = 
           (10,10))
plt.scatter(feature_fraudes.Amount , feature_fraudes.Time)
plt.title('Visualisation')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()

#Matrice de corrélation
corr  = feature.corr()
plt.figure(figsize=(10 , 8))
sns.heatmap( corr ,  cmap ='YlGnBu')
plt.show()



# %% [markdown]
# Modélisation 

# %%
#Séparation des données en valeurs d'entré et valeur de sortie  
y =  np.array(feature['Class'])
x = feature.drop('Class' , axis = 1)
feature_list = list(x.columns)
x = np.array(x)
print(x)
from sklearn import model_selection
from sklearn.metrics import precision_recall_curve, average_precision_score
x_train , x_test ,  y_train , y_test = model_selection.train_test_split(x , y , test_size = 0.3)
from sklearn import  preprocessing
std_scaler = preprocessing.StandardScaler().fit(x_train) # Apprend à standardisé les vecteurs de x pour avoir la meme échelle
x_train_std = std_scaler.transform(x_train)
x_test_std = std_scaler.transform(x_test)
from sklearn import svm
classifier = svm.SVC(kernel = 'rbf' , gamma = 0.01)#Noyau gaussien
#Entrainte le classifier sur les jeux de donnée
classifier.fit(x_train_std , y_train)
y_test_pred = classifier.decision_function(x_test_std)#La prédiction se fait sur le jeu de test
#Courbe Précison-rappel
precision , recall , thresholdes  =  precision_recall_curve (y_test , y_test_pred ) # nombre de fraude ; Capacité du modèle à trouver les TP
average_precision = average_precision_score(y_test , y_test_pred ) #Air sous la courbe plus le score est proche de 1 plus le modèle est efficace 

# Tracé
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Courbe PR (AP = {average_precision:.2f})', color='blue')
plt.xlabel('Recall (Rappel)', fontsize=14)
plt.ylabel('Precision (Précision)', fontsize=14)
plt.title('Courbe Précision-Rappel (SVM)', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.show()









# %% [markdown]
# L'axe horizontal Recall permet de savoir parmit les fraudes détecté combien le modèle en a déterminé ;Recall =TP/TP+FN
# L'axe vertical permet de savoir parmi les prédictions de fraude faites par le modèle, combien étaient vraiment des fraudes ; Precision = TP/TP
# +FP
# 
# Mon modèle atteint une average precision (précision moyenne) de 0,84 pour détecter les fraudes bancaires.
# 


