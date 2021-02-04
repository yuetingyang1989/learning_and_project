# Yueting's Hachathon jupyter notebooks (Dimension Reduction - UMAP)

* required library: 
conda install -c conda-forge umap-learn

* data source :
reviews_filtered.txt

There are 3 notebooks:
* review_basic is to inspect vocabulary and word count distribution, which is used to find a threshold to filter out too short reviews.

* copy_embedding is to copy the tensorflow version of processing text data into batches and transfer them to universal embeddings. I intend to directly apply umap here, but failed with tensorflow datatype manipulation.

* embedding_validation_test is to test :
    * if universal sentence encoding is able to find top K similar text as the input text
    * if UMAP is able to reduced the dimension
        * tried recommended components with 15 - 20, but due to the small amout of manually made sample, I can only put to 5, which is too little
        * tried to visualize the first 2 dimensions after UMAP, which is not ideally to seperate different tags
        
* UMAP is to test dimension reduction:
    * import 10k text samples
    * UMAP has to take all the embeddings into the model to project them into a reduced embeding space
    * process universal encoding 512 dimensions to 384 dimensions in 1 min
    * generate the top 5 similar text from input text from both models (USE & UMAP)
    * compare the output text, which are partially overlapped 

