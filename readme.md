### Recommendation System Using Collaborative Filtering

Collaborative Filtering can be divided into following two categories:
     
**1. Memory-based collaborative filtering**     
The memory based approach can be further divided into user-based similarity method and item-based method. In memory-based methods, we make recommendations by considering similarity between users or items. In user-based methods, for predicting rating given by the user to an item, cosine-similarity is used to find similar users and then their ratings given to the item are averaged out. Likewise for item-based approaces, items closer in space to the correct item are found out and recommended.
      
**2. Model-based collaborative filtering**     
The model based approach are used to find out latent factors involved in the interaction between users and items. To find out these latent factors, we use matrix-factorization methods such as [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) and Principal Component Analysi. This handles [sparsity](https://en.wikipedia.org/wiki/Sparse_matrix) of the rating matrix well.   
       
       
#### Hybrid Based Recommender System
       

#### Problems in collaborative filtering      