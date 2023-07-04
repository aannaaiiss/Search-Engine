import os
from collections import Counter,defaultdict #O(n)
import re
import spacy 
import math
from io import open
nlp = spacy.load('fr_core_news_md')
import nltk
from nltk.corpus import stopwords

#main
def main(BDOC,query):
    #On initialise correctement les valeurs entrées en paramètre ainsi que notre dictionnaire finale qui nous permettra de comparer les similarités entre elles
    query = vectorize_query(query)
    title2sim ={}

    #indexation : on crée un dictionnaire qui associe à un mot les documents dans lequel il est présent ainsi que son occurence
    BDOC2 = open(BDOC, mode='r')
    tok2doc2occ = indexation(BDOC2)
    
    #On crée le sous-ensemble des documents pertinents
    set_docs_pertinents = set({})
    for word in query :
            docs_pertinents = set(tok2doc2occ[word].keys())
            set_docs_pertinents.update(docs_pertinents)
            
    #On parcourt la BDOC pour récupérer les documents pertinents et créer leur vecteur
    BDOC2 = open(BDOC, mode='r')
    for doc in BDOC2:
        title = find_title(doc)
        nom_doc = find_nom_doc(doc)
        if nom_doc in set_docs_pertinents:
            
            #On nettoie et on lemmatize le document pour récupérer le nombre de mots dans le document pour le TF-IDF
            doc = clean(doc)       
            doc = lemmatize(doc)

            #On assigne les valeurs suivantes à ces deux variables pour les passer en paramètre des fonctions suivantes
            longueur_doc = len(doc)
            longueur_set_docs_pertinents = len(set_docs_pertinents)
                               
            #On crée le vecteur du document : ici il s'agit d'un dictionnaire qui assigne à un mot son score TF-IDF. Pour calculer le score TF-IDF on a besoin du dico de l'indexation pour accéder à l'occurence du mot dans le document
            vector_doc = vectorize(doc,nom_doc,longueur_doc,tok2doc2occ,longueur_set_docs_pertinents)
 
            #On calcule sa similarité avec la query
            similarity = similarity_cos(vector_doc,query)

            #On associe dans un dictionnaire chaque titre de document à sa similarité pour pouvoir ensuite les trier
            title2sim[title]=similarity
    #Tri
    results = make_list(title2sim)

    return affichage_results(results,BDOC)

#Fonctions principales
def indexation(BDOC):
    tok2doc2occ = defaultdict(dict)
    stops = set(stopwords.words('french'))
    for doc in BDOC:
        nom_doc = find_nom_doc(doc)
        doc = clean(doc)       
        doc = lemmatize(doc)
        
        #On crée pour chaque document un dictionnaire qui associe aux mots qui le composent leur occurence.
        tok2occ = Counter([token for token in doc if (token != ' ' and token not in stops)])
        for token in doc:
            if token not in stops :

                #On associe ensuite le dictionnaire en question à l'identifiant du document pour pouvoir le parcourir pendant la phase d'interrogation
                tok2doc2occ[token][nom_doc] = tok2occ[token]
    return tok2doc2occ


def vectorize(doc,nom_doc,longueur_doc,tok2doc2occ, longueur_set_docs_pertinents):
    stops = set(stopwords.words('french'))
    tok2TFIDF = {}

    #A nouveau, on crée le même tok2occ que plus haut (même si ça fait doublon) pour pouvoir itérer sur les mots qui composent le document et avoir accès à leur occurence
    tok2occ = Counter([token for token in doc if (token != ' ' and token not in stops)])
    for word in tok2occ:

        #TF = nombre d'occurence du mot / nombre total de mots dans le document
        TF = tok2occ[word]/longueur_doc

        #IDF = log(nombre de documents contenant le mot/nombre de documents du sous-ensemble de documents (les documents pertinents uniquement)
        docs_pertinents_word = tok2doc2occ[word].keys()
        IDF = math.log(len(docs_pertinents_word)/longueur_set_docs_pertinents)

        tok2TFIDF[word] = TF * IDF
    return tok2TFIDF

def similarity_cos(vector_doc, query):
    norme_doc = 0
    produit_scalaire = 0
    norme_query = 0

    #On itère sur les éléments des deux vecteurs pour créer la somme de leurs carrés 
    for element in vector_doc :
        norme_doc += vector_doc[element]**2
    for element in query :
        norme_query += query[element]**2

    #Pour calculer le produit scalaire, on regarde les clés en commun des deux dictionnaires car ils correspondent à la position i des vecteurs et on effectue la somme des produits
        if element in vector_doc :
            produit_scalaire += (vector_doc[element]*query[element])

    #On rapporte les normes à leur racine    
    norme_query = math.sqrt(norme_query)
    norme_doc = math.sqrt(norme_doc)
    
    if (norme_doc*norme_query) != 0 :

        #On retourne la valeur absolue de la similarité cosinus car elle peut être négative
        return math.sqrt(((produit_scalaire/(norme_doc*norme_query)))**2)
    return -2
    
def make_list(title2sim):
    
    #On trie le dictionnaire qui associe un titre de document à sa similarité cosinus en fonction de la similarité
    sortdict = sort_dico(title2sim)

    #On crée la liste des résultats à afficher en récupérant les clés triées du dictionnaire 
    results = [title for title in sortdict.keys()]
    return results


#Fonctions auxiliaires
def lemmatize(doc):
    tokenized_doc = nlp(doc)
    tokenized_doc = [token.lemma_ for token in tokenized_doc]
    return tokenized_doc

def clean(doc):
    #On efface la balise et on supprime la ponctuation du document 
    doc = re.sub("<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", "", doc).strip().lower()
    doc = re.sub(r'[^\w\s]', ' ', doc) 
    return doc

def find_nom_doc(doc):
    
    #On cherche l'identifiant du document grâce à une regex et on l'assigne à une variable qu'on retourne
    baliseXML = re.search("^<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", doc)
    nom_doc = re.search("[0-9]{7}", doc)
    if nom_doc == None :
        nom_doc = re.search("[0-9]{6}", doc)
    nom_doc = nom_doc.group() 
    return nom_doc

def find_title(doc):

    #Même chose mais pour le titre du document
    title = re.search("<title>.*</title>", doc)
    if title != None :
        title = title.group()
        title = title.replace("<title>","").replace("</title>","")
    else :
        title = find_nom_doc(doc)
    return title

def vectorize_query(query):
    stops = set(stopwords.words('french'))
    query = lemmatize(query)
    new_query ={}
    for word in query :
        if word not in stops :
            new_query[word] = new_query.get(word,0)+1
    return new_query

def sort_dico(title2sim):
    #On effectue le tri par ordre de décroissance
    title2sim_list = reversed(sorted(title2sim.items(), key=lambda x:x[1]))
    return dict(title2sim_list)

#Fonctions d'affichage
def affichage_results(results,BDOC):
    print(f"{len(results)} results found: ")
    i = 0
    debut = 0

    #Si aucun résultat n'est trouvé :
    if results == []:
        return "Sorry, no results"
    else:

        #Si le nombre de résultats trouvés est inférieur à 10
        if len(results)<10:
            for element in results:
                print(element)
                input2 = input("Please, copy the title of the document you're interested in : ")
                affichage_doc(input2,BDOC)
                return ""
        else :

            #Sinon, on affiche les résultats par tranche de 10 et on demande à l'utilisateur s'il veut afficher la suite. Pour avoir accès au document qui l'intéresse, l'utilisateur doit copier le titre du document dans l'input.
            for j in range(10,len(results),10):
                for element in results[debut:j]: 
                  print(element)
                i += 1
                debut += 10
                print(f"Page {i}")
                input1 = input("Show more? (Y/n) ")
                if input1 == "no" or input1 == "No" or input1 == "n" or input1 == "N":
                  input2 = input("Please, copy the title of the document you're interested in : ")
                  affichage_doc(input2,BDOC)
                  return ""
                if input1 == "yes" or input1 == "YES" or input1 == "y" or input1 == "Y":
                  continue
                else :
                    while input1 != "no" and input1 != "No" and input1 != "n" and input1 != "N" and input1 != "yes" and input1 != "YES" and input1 != "y" and input1 != "Y":
                      input1 = input("Show more? (Y/n)")

            #On affiche la dernière page
            for element in results[j:]:
              print(element)
            print("No more results")

            #Si aucun document ne l'intéresse, l'utilisateur doit arrêter manuellement le programme
            input3 = input("Please, copy the title of the document you're interested in, exit otherwise : ")
            affichage_doc(input3,BDOC)
        return ""


def affichage_doc(input1,BDOC):
    BDOC2 = open(BDOC,mode='r')

    #On parcourt la BDOC pour retrouver le fichier à afficher à l'utilisateur
    for doc in BDOC2 : 
        if input1 in doc :
            doc = re.sub("<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", "", doc)
            print(doc)
            return""

    #Si l'utilisateur se trompe, il doit copier à nouveau un titre
    print("No document found")
    input2 = input("Please, copy the title of the document you're interested in : ")
    return affichage_doc(input2, BDOC)
            
BDOC = "OD1"
query = input("[Enter your query here] ")
print(main(BDOC,query))

#Code pour la métrique d'évaluation (ne marche pas, nous n'avons pas trouvé de solution)

def find_requete(doc):
    doc = open("documents_metriques/"+doc, mode='r', encoding='utf-8')
    for line in doc :
        if "<suj>" in line :
            requete = re.search("<suj>.*<\/suj>", line)
            requete = requete.group()
            requete = requete.replace("<suj>","").replace("</suj>","")
            requete = lemmatize(requete)
        else :
            continue
    return requete

def find_cepts(doc):
    stops = set(stopwords.words('french'))
    liste_cepts=[]
    doc = open("documents_metriques/"+doc, mode='r', encoding='utf-8')
    for line in doc :
        if '<c>' in line :
            cept = re.search("<c>.*</c>", line)
            cept = cept.group()
            cept = cept.replace("<c>","").replace("</c>","")
            cept = set(lemmatize(cept)).difference(stops)
        else :
            continue
        liste_cepts.extend(cept)
    return liste_cepts

def metrique(document, BDOC):
    requete2docs = defaultdict(dict)
    docs = []
    requete = find_requete(document)
    liste_cepts = find_cepts(document)

    #On parcourt la BDOC pour rechercher tous les documents qui contiennent au moins un des mots lemmatisés compris entre les balises <c>
    for doc in BDOC :
        for word in liste_cepts:
            if word in doc :
                title = find_title(doc)
                docs.append(title)
                
        #On associe à une requête une liste de documents qui contiennent au moins un mot des expressions comprises entre les balsies <c>
        requete2docs[requete] = docs
    return requete2docs

#On crée au préalable 5 documents qui contiennent les 5 premières <record> du fichier OT1
for fichier in os.listdir("."): 
    print(metrique(fichier, BDOC))

