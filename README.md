This is a course project that implements "Approximate DBSCAN under Differential Privacy" with a minimal version. 

For each synthetic dataset: moon, circle, and blobs, we have an individual ipynb file to test. 

We conduct grid search for parameter optimization, but the visualization does not look good with high AMI and ARI,
and therefore we mannually pick parameters episolon and T for visualization. 

Though the final visualization does not come along with high AMI and ARI scores, but the spans roughly cover

the regions, which demonstrate the effectiveness of the algorithm. 

Crash dataset available: https://github.com/QiuTedYuan/DpDBSCAN/blob/master/datasets/crashes_240928.zip
