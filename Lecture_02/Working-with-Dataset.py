import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

ds = pd.read_csv('./movie_metadata.csv')
print type(ds)
print len(ds.columns)

data = ds.values
print type(data)
print data.shape
#ds.head(n=5)
ds.tail(n=3)


#for ix in ds.columns:
   #print ix
print type(ds.columns)

titles = ds.get('movie_title')
print len(titles)
print type(titles)

count = 0
for ix in titles:
    if count<=10:
        print ix
    count+=1


#The above written code should return the following . 
#Avatar 
#Pirates of the Caribbean: At World's End 
#Spectre 
#The Dark Knight Rises 
#Star Wars: Episode VII - The Force Awakens             
#John Carter 
#Spider-Man 3 
#Tangled 
#Avengers: Age of Ultron 
#Harry Potter and the Half-Blood Prince 
#Batman v Superman: Dawn of Justice 


len_titles = []
for ix in titles:
    len_titles.append(len(ix))
#print len(len_titles)
len_titles = np.array(len_titles)
print len_titles.max()
print len_titles.min()
print len_titles.mean()
