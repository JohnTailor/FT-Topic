from sklearn.datasets import fetch_20newsgroups
import os
import numpy as np




if __name__ == "__main__":
    #import matplotlib
    #matplotlib.use('TKAgg') #only needed if work on a remote server to show plots

    print("Fine-tuning sentence enocder on 20News")

    from finetuneTM import fttopic
    docs = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data #get raw data
    folder = "outputs/"
    fine_tuned_modelname=folder+"tunedEncoder"
    fttopic.getFTTopic(docs, modelToTune="all-MiniLM-L6-v2", outputName=fine_tuned_modelname, margin=0.16, ftri=0.24, fpos=0.08, nneg=1, ep=4, trainingFrac=1, device="cuda")

    print("Run topic model using fine-tuned encoder")
    from finetuneTM import senClu
    topic_model= senClu.SenClu(encoder=fine_tuned_modelname)
    topics, probs = topic_model.fit_transform(docs, nTopics=20,nEpochs=1,loadAndStoreInFolder=folder)

    topic_model.saveOutputs(folder)  # Save outputs in folder, i.e. csv-file and visualizations

    for it,t in enumerate(topics): #Print Topics
        print("Topic",it,str(t[:10]).replace("'",""))

    print("Top 10 words with scores for topic 0", topic_model.getTopicsWithScores()[0][:10])
    print("Distribution of topics for document 0", np.round(topic_model.getTopicDocumentDistribution()[0],3))
    print("Distribution of topics", np.round(topic_model.getTopicDistribution(), 3))
    print("First 4 sentences for top doc for topic 0 ", topic_model.getTopDocsPerTopic()[0][0][:4])
    print("Top 3 sentences for topic 0 ", topic_model.getTopSentencesPerTopic()[0][:3])

    if not os.path.exists("visual.py"):
        print("To create interactive visualization in new browser window, download 'visual.py' https://github.com/JohnTailor/BertSenClu/blob/main/visual.py and put it in the same directory")
    else:
        import subprocess
        print("Optional: Launching visualization in browser from stored data (can also be called from Shell)")
        subprocess.run("streamlit run visual.py -- --folder "+folder,shell=True)