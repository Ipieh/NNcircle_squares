#%%
import matplotlib.pyplot as plt
import PIL.Image as img
import numpy as np

#%%
def sigmoide(x):
         return 1/(1+np.exp(-x))
        
def derivee_sigmoide(x):
        return x*(1-x)
        #définition de la fonction d'activation et sa dérivée
#%%
def traite_image(nom_image):
    mat_linear=[]
    photo=img.open(nom_image)
    photo.thumbnail((8,8))
        
    matrice=np.asarray(photo.convert('L'))
    
    for ligne in matrice:
        for colonne in ligne:
            mat_linear.append([colonne/255])
    
    return np.array(mat_linear).T

#ce programme fait d'une image, une matice carree 8*8 puis une matrice ligne.
# suivant l'ordre d'une ligne (ligne1,ligne2,ligne3...) format (1,64)
#%%
class couche():
    
    def __init__(self,entrees,nb_neur):
        
        
        self.biais=np.random.rand()
        self.s_weights= np.random.random((entrees,nb_neur)) 
        
class Reseau():
    
    def __init__(self,couche1,couche2,couche3):
        self.couche1=couche1
        self.couche2=couche2
        self.couche3=couche3
        

        
        self.learning_rate=0.01
        

        
    def predit_intermediaire (self,nom_image):
        
        t_inputs= traite_image(nom_image)
        
        
        resultat1= sigmoide((np.dot(t_inputs,self.couche1.s_weights))/64+self.couche1.biais)
        
        #la sortie d'un neurone est une moyenne pondérée des entrées. matriciellement, toutes les sorties de neurones
        #forment une matrice ligne. On la multiplie par une autre matrice pour la deuxieme couche ou: sortie est
        #en ligne: les coefficients correspondant a chauqe poids synaptique pour un meme neuronne j 
        #sont donc sur la jeme colonne (entrees,nb de neuronnes)
        
        resultat2=sigmoide((np.dot((resultat1),self.couche2.s_weights))/64+self.couche2.biais)
        
        resultat3=sigmoide((np.dot((resultat2),self.couche3.s_weights))/64+self.couche3.biais)
        
        return t_inputs,resultat1,resultat2,resultat3
        
        #calcul de chaque neurone: poids synaptiques*ce qu'elle reçoit, sigmoidisé pour rendre 1 ou 0 -> []
        
    def apprendre(self,entrees,attendu,n):
            
        
         for i in range(n):
            for j in range(len(entrees)):
                entree,r1,r2,r3=self.predit_intermediaire(entrees[j])
                erreur=attendu[j]-r3 
                
                chaine_conservee=2*erreur*derivee_sigmoide(r3)
                correction3= np.dot(r2.T, 2*erreur*(derivee_sigmoide(r3)))
                self.couche3.s_weights+= self.learning_rate*correction3
                self.couche3.biais+=chaine_conservee
                
    
                chaine_conservee=np.dot(chaine_conservee,self.couche3.s_weights.T)*derivee_sigmoide(r2)
                correction2=np.dot(r1.T,chaine_conservee)
                self.couche2.s_weights+= self.learning_rate*correction2
                self.couche2.biais+=chaine_conservee
                

                chaine_conservee=np.dot(chaine_conservee,self.couche2.s_weights.T)*derivee_sigmoide(r1)
                correction1=np.dot(entree.T,chaine_conservee)
                self.couche1.s_weights+= self.learning_rate*correction1
                self.couche1.biais+=chaine_conservee
            
            
                
    def predit(self,nom_image):
        
        hidden1,hidden2,hidden3,output=self.predit_intermediaire(nom_image)
        
        print("carre:",output[0,0])
        print("triangle:",output[0,1])
        print("cercle:",output[0,2])

    def apprentissage_courbe(self,entrees,attendu,n,taux_apprentissage):

        self.learning_rate = taux_apprentissage
        norme_erreur=[]

        for i in range(n):
            moyenne=0
            for j in range(len(entrees)):
                entree,r1,r2,r3=self.predit_intermediaire(entrees[j])
                erreur=attendu[j]-r3 

                for k in erreur[0]:
                    moyenne+=abs(k)/(len(erreur))
                
            
            

                chaine_conservee=2*erreur*derivee_sigmoide(r3)
                correction3= np.dot(r2.T, 2*erreur*(derivee_sigmoide(r3)))
                self.couche3.s_weights+= self.learning_rate*correction3
                self.couche3.biais+=chaine_conservee
                
    
                chaine_conservee=np.dot(chaine_conservee,self.couche3.s_weights.T)*derivee_sigmoide(r2)
                correction2=np.dot(r1.T,chaine_conservee)
                self.couche2.s_weights+= self.learning_rate*correction2
                self.couche2.biais+=chaine_conservee
                

                chaine_conservee=np.dot(chaine_conservee,self.couche2.s_weights.T)*derivee_sigmoide(r1)
                correction1=np.dot(entree.T,chaine_conservee)
                self.couche1.s_weights+= self.learning_rate*correction1
                self.couche1.biais+=chaine_conservee
            
            moyenne=moyenne/len(entrees)
            norme_erreur.append(moyenne)

        X1=[j for j in range(n//2)]
        X2=[j for j in range ((n//2+1),n)]
        plt.plot(X1,norme_erreur[:n//2])
        plt.plot(X2,norme_erreur[(n//2)+1:])
        #plt.title("taux d'apprentissage: "+str(taux_apprentissage))
        
        plt.show()

#%%            
#################
## FORMATTAGE  ##
#################
def ajouter_loc(liste):
    for i in range (len(liste)):
        liste[i]="Users/pierina/Desktop/programs/figures/"+liste[i]
        
    return liste         

#sortie attendue, elle est en ligne. Ici on formatte la base d'apprentissage, 
# #d'abord carre, puis triangle et finalement cercle.
#on verra plus tard pour lordre d'apprentissage aléatoire
def sorties():
    sortie=[]
    outputs=[[1,0,0],[0,1,0],[0,0,1]]
    #outputs[i] est de forme (3,)
    #[1,0,0] est un carre, [0,1,0] est un triangle, [0,0,1] est un cercle
    for i in range(3):
        for j in entrees[i]:
            sortie.append(outputs[i])
    return np.array(sortie)

#%%
######################
# CREATION DU RESEAU #
######################
#initialisation des couches. !!!!À étudier, l'influence du nombre de couches pour mon etude. !!!! si te sobra la energía
c1=couche(64,64)
c2=couche(64,64)
c3=couche(64,3)

Reseau1=Reseau(c1,c2,c3)

#%%

carres=["carre.png","carre(1).png","carre(2).jpeg","carre(3).png","carre(4).png","carre(5).jpeg"]
triangles=["triangle.jpeg","triangle(1).jpeg","triangle(2).jpeg","triangle(3).jpeg","triangle(4).jpeg","triangle(5).png"]
cercles=["cercle(1).png",'cercle(2).png','cercle(3).png','cercle(4).png']




#carres,triangles,cercles=ajouter_loc(carres),ajouter_loc(triangles),ajouter_loc(cercles)

entrees=[carres,triangles,cercles]
figures=carres+triangles+cercles
resultats=sorties()
#%%
Reseau1.apprentissage_courbe(carres+cercles,[[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],500,0.07)

# %%
Reseau1.apprendre(["carre.png","cercle(1).png"],[[1,0,0],[0,0,1]],7000)# %%

# %%
