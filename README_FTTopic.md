
# FT-Topic - Fine-tuning LLM encoder for topic modeling

**FT-Topic** fine-tunes an LLM encoder (such as BERT) to achieve better topic modeling outcomes. It does so in an unsupervised manner. It can be used jointly with any topic modeling technique that relies on embeddings, e.g., BerTopic and SenClu. 

We evaluate it using SenClu, which is also included in this Repo.

For details on FTTopic see: To cite FTTopic use (to appear)
For details on SenClu see: To cite FTTopic use (to appear)


    
## Quick Start
We start by extracting topics from the 20 newsgroups dataset containing English documents. We first fine-tune a sentence-encoder model and store it.

```python
from finetuneTM import fttopic    
from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data #get raw data
folder = "outputs/"
fine_tuned_modelname=folder+"tunedEncoder"    
fttopic.getFTTopic(docs, modelToTune="all-MiniLM-L6-v2", outputName=fine_tuned_modelname, margin=0.16, ftri=0.24, fpos=0.08, nneg=1, ep=4, trainingFrac=1, device="cuda")
```


Now, we use the fine-tuned model for topic modeling:

```python
from finetuneTM import senClu
topic_model= senClu.SenClu(encoder=fine_tuned_modelname)
topics, probs = topic_model.fit_transform(docs, nTopics=20, loadAndStoreInFolder=folder)


```

After generating topics and their probabilities, we can save outputs:

```python
>> topic_model.saveOutputs(folder) #Save outputs in folder, i.e. csv-file and visualizations
```

and look at topics

```python


>>for it,t in enumerate(topics):
    print("Topic",it,t[:10])
    
Topic 0 [encryption, key, ripem, privacy, rsa, clipper, encrypted, escrow, nsa, secure]
Topic 1 [government, militia, amendment, federal, law, constitution, firearm, regulated, administration, clinton]
Topic 2 [launch, satellite, lunar, space, orbit, mission, spacecraft, larson, probe, shuttle]
Topic 3 [patient, hiv, disease, infection, candida, vitamin, antibiotic, diet, symptom, smokeless]
...

 ```  

We can also use an interactive tool for visualization and topic analysis that runs in a browser. It can be called command line with the folder containing topic modeling outputs:

You need to **download** the [**visual.py**](https://github.com/JohnTailor/BertSenClu/blob/main/visual.py) from the repo first

```console
streamlit run visual.py -- --folder "modelOutputs/"
```

It can also be called from python:

```python
import subprocess
folder = "modelOutputs/"
subprocess.run("streamlit run visual.py -- --folder "+folder,shell=True)
```

The interactive visualization looks like this:

<img src="https://github.com/JohnTailor/BertSenClu/blob/main/images/visual.PNG" width="100%" height="100%" align="center" />

If you scroll down (or look into the folder where you stored outputs), you see topic relationship information as well, i.e., a TSNE visualization and a hierarchical clustering of topics:

<img src="https://github.com/JohnTailor/BertSenClu/blob/main/images/topic_visual_hierarchy.png" width="60%" height="60%" align="center" />
<img src="https://github.com/JohnTailor/BertSenClu/blob/main/images/topic_visual_tsne.png" width="60%" height="60%" align="center" />


We can also access outputs directly by accessing functions from the model

```python

>> print("Top 10 words with scores for topic 0", topic_model.getTopicsWithScores()[0][:10])
Top
10
words
with scores for topic 0[('encryption', 11.269135), ('key', 11.173454), ('ripem', 10.151058), ('privacy', 10.070835), ('rsa', 7.3271174), ('clipper', 6.8211393), ('encrypted', 6.567956), ('escrow', 5.993511), ('nsa', 5.853071), ('secure', 5.4898496)]

>> print("Distribution of topics for document 0", np.round(topic_model.getTopicDocumentDistribution()[0], 3))
Distribution
of
topics
for document 0[0. 0. 1....0. 0. 0.]

>> print("Distribution of topics", np.round(topic_model.getTopicDistribution(), 3))
Distribution
of
topics[0.022
0.061
0.024
0.026
0.067
0.079
0.155
0.043
0.061
0.039
0.031
0.198
0.018
0.033
0.033
0.012
0.016
0.029
0.033
0.02]

>> print("First 4 sentences for top doc for topic 0 with probability and ", topic_model.getTopDocsPerTopic()[0][0][:4])
First
4
sentences
for top doc for topic 0 (['[...]>\n', '[...]>\n\n', "If the data isn't there when the warrant comes, you effectively have\n", 'secure crypto.  '], 1.0, 8607)

>> print("Top 3 sentences for topic 0 ", topic_model.getTopSentencesPerTopic()[0][:5])
Top
3
sentences
for topic 1[
    ('enforcement.\n\n    ', 0.22597079), ('Enforcement.  ', 0.22597079), ('to the Constitution.\n\n   ', 0.22434217)]
# The sentences show that the sentence partitioning algorithm used is not the best... (It splits based on carriage returns. Still topic modeling results are good. It's also easy to use another one, or preprocess the data    

```


## How it works
The steps for topic modeling with **Bert-SenClu** are
<ol>
  <li>Splitting docs into sentences</li>  
  <li>Embedding the sentences using pretrained sentence-transformers</li>
  <li>Running the topic modeling</li>
  <li>Computing topic-word distributions based on sentence to topic assignments</li>
</ol>
The outcomes of the first two steps are stored in a user-provided folder if parameter "loadAndStoreInFolder" is set explicitly in "fit_transform". By default this is not the case (i.e., "loadAndStoreInFolder"=None).  **Bert-SenClu** can reuse the stored precomputed sentence partitionen and embeddings, which speeds up re-running the topic modeling, e.g., if you want to change the number of topics. However, if you alter the data, you need to delete the folder, i.e., the files with the precomputed sentence embeddings and partitionings.  
 
You can change each algorithm in these steps, especially the algorithm for sentence partitioning as well as the pre-trained sentence embedder. As you saw in the example, the used algorithm for sentence partitioning is not that great for the newsgroup dataset, but the overall result is still good.

The (main) function "fit_transform" has a hyperparameter "alpha" (similar to other models like LDA), which guides the algorithm on how many topics a document should contain. Setting it 0, means that a document likely has few topics. Setting it to 1 (or larger) means it is more likely to have many (for longer documents). As default, you can use 0.5/sqrt(nTopics). 


## Citation
To cite FTTopic use (to appear)

To cite the [Bert-SenClu Paper](https://arxiv.org/abs/2302.03106), please use the following bibtex reference:

```bibtex
@article{schneider23,
  title={Efficient and Flexible Topic Modeling using Pretrained Embeddings and Bag of Sentences},
  author={Schneider,Johannes},
  journal={arXiv preprint arXiv:2302.03106},
  year={2023}
}
```
