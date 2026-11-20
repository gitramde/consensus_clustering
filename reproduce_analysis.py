import sys,time,json
sys.path.insert(0,'source')
import numpy as np
from data_extraction import data_load
from data_preparation import process
from clustering import kmeans_clustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage,fcluster
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=2):
 t=time.time(); d,c=data_load().get_data(); x=process().vectorize_data(d.data); print('data',len(d.data),x.shape,flush=True)
 x=TruncatedSVD(n_components=200,random_state=42).fit_transform(x); parts=kmeans_clustering(x,len(c)); print('base done',time.time()-t,flush=True)
 n=len(parts); sim=np.array([[adjusted_rand_score(a,b) for b in parts] for a in parts]); med=np.argmax(sim.sum(axis=1)); co=np.zeros((len(x),len(x)),dtype=np.float64)
 for labels in parts: co += labels[:,None]==labels[None,:]
 co/=n; dist=1-co; np.fill_diagonal(dist,0); labels=fcluster(linkage(squareform(dist),method='average'),len(c),criterion='maxclust')
 metrics=lambda y: {'ari':adjusted_rand_score(d.target,y),'nmi':normalized_mutual_info_score(d.target,y),'ami':adjusted_mutual_info_score(d.target,y)}
 out={'samples':len(x),'class_counts':np.bincount(d.target).tolist(),'tfidf_shape':list(process().vectorize_data(d.data).shape),'base':[metrics(p) for p in parts], 'representative_index':int(med),'representative':metrics(parts[med]),'coassociation':metrics(labels),'seconds_total':time.time()-t,'sklearn_version':__import__('sklearn').__version__}
 with open('reproduced_results.json','w') as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2),flush=True)
