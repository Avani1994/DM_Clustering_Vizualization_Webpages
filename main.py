
# coding: utf-8

# In[2]:

from os import listdir
from bs4 import BeautifulSoup
import urllib2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# What to be done with test file ??

train_file = open('webkb-train-stemmed.txt', "r").read()
test_file = open('webkb-test-stemmed.txt', "r").read()

train_file = train_file.split('\n')
test_file = test_file.split('\n')
train_file_new  =[]
test_file_new = []
for line in train_file:
    words = line.split()
    #print words[0]
    if(len(words)>1):
        words.remove(words[0])
    #print words
    line = ' '.join([word for word in words])
    train_file_new.append(line)
for line in test_file:
    words = line.split()
    #print words[0]
    if(len(words)>1):
        words.remove(words[0])
    #print words
    line = ' '.join([word for word in words])
    test_file_new.append(line)  
print len(test_file_new)


print train_file_new[len(train_file_new)-1]
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 1000) 

train_data_features = vectorizer.fit_transform(train_file_new[:-1])
#test_data_features = vectorizer.fit_transform(test_file)
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print train_data_features

vocab = vectorizer.get_feature_names()

#print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print count, tag

test_data_features = vectorizer.transform(test_file_new[:-1])

print test_data_features.toarray()


# In[4]:

from sklearn.cluster import KMeans
kmeans =KMeans(n_clusters=4, random_state=0, init = 'k-means++').fit(train_data_features)



# In[5]:

kmeans.cluster_centers_
np.set_printoptions(threshold = 'nan')
kmeans.labels_


# In[6]:

tcounts = [0,0,0,0]
for tvector in train_data_features:
    label = kmeans.predict(tvector.reshape(1,-1))
    tcounts[label[0]] +=1
print tcounts


# In[7]:

from sklearn.cluster import AffinityPropagation
af = AffinityPropagation(damping =0.99,max_iter = 75).fit(train_data_features)
len(af.cluster_centers_)
#af.labels_


# In[14]:

len(af.cluster_centers_)
unique, counts = np.unique(af.labels_, return_counts=True)
dict(zip(unique, counts))


# In[9]:

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# normalize dataset for easier parameter selection
#X = StandardScaler().fit_transform(X)

# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(train_data_features, quantile=0.3)

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(train_data_features, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)


ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

two_means = cluster.MiniBatchKMeans(n_clusters=4,init='random')

ward = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward', connectivity=connectivity)

spectral = cluster.SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity="nearest_neighbors")

dbscan = cluster.DBSCAN(eps=.2)

affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)

average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=4, connectivity=connectivity)

birch = cluster.Birch(n_clusters=4)

clustering_algorithms = [two_means, spectral, ward, average_linkage, dbscan, birch]

clustering_names = ['MiniBatchKMeans',
    'SpectralClustering', 'Ward', 'AverageLinkage'
    'DBSCAN', 'Birch']

#'MeanShift'
# ms
# 'AgglomerativeClustering'
#






# In[17]:

for name, algorithm in zip(clustering_names,clustering_algorithms):
    print name
    algorithm.fit(train_data_features)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
        unique, counts = np.unique(labels, return_counts=True)
        print dict(zip(unique, counts))

        
        
        


# In[64]:

#print train_data_features

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.01 * (1 - .01)))
sel.fit_transform(train_data_features)

print sel


# In[24]:

from sklearn.decomposition import PCA

nf = 100
pca = PCA(n_components=nf)
# X is the matrix transposed (n samples on the rows, m features on the columns)
pca.fit(train_data_features)

X_new = pca.transform(train_data_features)


# In[25]:

for name, algorithm in zip(clustering_names,clustering_algorithms):
    print name
    algorithm.fit(X_new)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
        unique, counts = np.unique(labels, return_counts=True)
        print dict(zip(unique, counts))


# In[5]:

import urllib2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RandomizedLasso
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
train_file_new  =[]
test_file_new = []
train_target = []
test_target = []
for line in train_file:
    words = line.split()
    #print words[0]
    if(len(words)>=1):
        if(words[0] == 'student'):
            train_target = train_target + [0]
        elif(words[0] == 'course'):
            train_target = train_target + [1]
        elif(words[0] == 'faculty'):
            train_target = train_target + [2]
        elif(words[0] == 'project'):
            train_target = train_target + [3]
        words.remove(words[0])
    #print words
    line = ' '.join([word for word in words])
    train_file_new.append(line)
for line in test_file:
    words = line.split()
    #print words[0]
    if(len(words)>1):
        words.remove(words[0])
        if(words[0] == 'student'):
            test_target = test_target + [0]
        if(words[0] == 'course'):
            test_target = test_target + [1]
        if(words[0] == 'faculty'):
            test_target = test_target + [2]
        if(words[0] == 'project'):
            test_target = test_target + [3]
    #print words
    line = ' '.join([word for word in words])
    test_file_new.append(line)  
#print len(test_file_new)



# In[6]:


vectorizers = TfidfVectorizer(min_df=1)
train_features = vectorizers.fit_transform(train_file_new[:-1]).toarray()
#print train_features






# In[7]:

#print train_file_new[len(train_file_new)-1]
#vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None, stop_words = None,max_features = 1000) 




#train_data_features = vectorizer.fit_transform(train_file_new[:-1])

#test_data_features = vectorizer.fit_transform(test_file)
# Numpy arrays are easy to work with, so convert the result to an 
# array
#train_data_features = train_data_features.toarray()

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(train_data_features).toarray()
print transformer.idf_


nf = 100
pca = PCA(n_components=nf)
# X is the matrix transposed (n samples on the rows, m features on the columns)
pca.fit(train_data_features)

X_new = pca.transform(train_data_features)


# In[ ]:

for name, algorithm in zip(clustering_names,clustering_algorithms):
    print name
    algorithm.fit(train_features)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
        unique, counts = np.unique(labels, return_counts=True)
        print dict(zip(unique, counts))
        label_counts =[[0,0,0,0] for y in range(4)]
        for cluster,label in zip(train_target,labels):
            label_counts[label][cluster] = label_counts[label][cluster]+1
        print label_counts


# In[ ]:



