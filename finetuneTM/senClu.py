#Python libs
import random,time
import numpy as np,os,pickle
import multiprocessing
#import sys
#sys.path.append('bertsenclu') #for visutils import

#PyTorch
import pandas as pd
import torch
import torch.cuda.amp as tca
import torch.nn as nn

#Other libs
from gensim import utils as gut #tokenizer
from sentence_transformers import SentenceTransformer #Sentence embedder see https://www.sbert.net/; pip install -U sentence-transformers
import pysbd #Sentence segmenter https://github.com/nipunsadvilkar/pySBD #pip install pysbd
from nltk.stem import WordNetLemmatizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #needed for huggingface transformer, otherwise get warning "The current process just got forked, Disabling parallelism to avoid deadlocks.. To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)"

class SenClu():
    def __init__(self,encoder='all-MiniLM-L6-v2',device='auto'):
        """
        Inputs:
        device...where to do computations : 'cpu' or 'cuda' (= on GPU) or 'auto' (= choose cuda if available else cpu)
        encoder...name of sentence encoder if
        """
        super(SenClu, self).__init__()
        self.maxTopWords = 50 #Top words per Topic computed
        self.nSenPerGroup =3
        self.device=device
        self.verbose=True
        self.ftname = encoder #'all-MiniLM-L6-v2'


    def kmeans_pp_init(self,X,k):
        """
        Compute kmeans++ initial clsuters
        #Inputs:
        #X... data tensor with n rows (samples) and d columns (features)
        #k...the number of centroids to choose
        #Outputs:
        #tensor of shape (d,k) containing the k initial centroids.
        """
        centroids = []
        n = X.shape[0]
        with tca.autocast():
            with torch.no_grad():
                # Choose first centroid randomly
                idx = torch.randint(n, size=(1,))
                r = torch.rand(k).to(self.device)
                centroids.append(X[idx[0]])
                # Choose the remaining centroids
                for i in range(1, k):
                    # Compute similiarity from data points to most similar centroid
                    D2 = torch.stack([torch.matmul(X,c) for c in centroids])
                    D2=torch.max(D2,dim=0)[0]
                    # Choose new centroid with probability proportional to similarity
                    probs = D2 / (torch.abs(torch.sum(D2))+1e-4)
                    cumulative_probs = torch.cumsum(probs, dim=0)
                    idx = torch.searchsorted(cumulative_probs, r[i-1])
                    idx=max(0,min(idx,n-1)) #this crashes otherwise with out of bounds, since add the division to 0 protection above (1e-4)
                    centroids.append(X[idx])
                cents=torch.stack(centroids,1)
        return cents



    def computeTopicModel_SenClu(self,docs, device, ntopics, nepoch=20, alpha=1,epsilon=0.001,cfg=None): #dimension of sentence embeddings # initial offset to add to topic to smoothen p(t|d) distribution; # alpha = final offset, serves also as prior to determine how many topics per document exist
        """
        Compute topic model given docs with sentence embeddings
        ---------------------------------------------
        Inputs:
        docs = documents as string
        device = cuda (GPU) or cpu
        ntopics = number of topics
        nepoch = number of training epochs
        alpha = parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document) , 0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha even if no sentence of document belongs to topic t
        epsilon = Convergence criterion: Stop computation if topic distribution changes by less than this for an epoch (even before nepoch have been executed)
        ----------------------
        Returns:
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentences;  ~ #docs x #sentences in doc
        """
        if self.verbose: print("Training topic model")
        self.device=device
        nd = len(docs)  # number of docs
        ioff= 8#cfg["ioff"] #initial smoothening of p(t|d), not very critical, if doc is very long, it should also be a bit larger istart =

        # Initialize topic vectors v_t randomly
        #embDims=len(docs[0][0]) # embDims = dimensions of pre-trained sentence encodings, usually 384
        # # Initialize topic vectors randomly
        # vec_t = (np.random.random((embDims, ntopics)) - 0.5)
        # sum_vt = np.sqrt(np.sum(vec_t ** 2, axis=0, keepdims=True))
        # vec_t = vec_t / (sum_vt + 1e-10)
        # vec_t = vec_t.astype(dtype=np.float32)

        closs = 0  # loss
        mloss = 0
        rloss0,closs0=0,0
        rloss, mrloss = 0, 0
        assignSen_ptd = [[] for _ in range(nd)]  # topic assignments of each sentence account for prob(t|d)
        #assignSen = [[] for _ in range(nd)]  # topic assignments of each sentence without prob(t|d)
        prob_t = [[] for _ in range(nd)]  # topic prob of assigned sentence
        with tca.autocast():
            ptd = torch.ones((ntopics, nd)) / ntopics  # p(t,d) uniform
            ptd = ptd.to(device)
            docs_torch = [torch.from_numpy(x.astype(np.float32)) for x in docs]
            docs_torch = [x.to(device) for x in docs_torch]
            iVecX=torch.cat([dx[np.random.choice(min(5,len(dx)),len(dx))] for dx in docs_torch])
            vec_t=self.kmeans_pp_init(iVecX,ntopics)
            # vec_t = torch.from_numpy(vec_t) #for random init
            # vec_t = vec_t.to(device)


        lalpha=2.0/len(docs)
        #if cfg["cos"]:
        with tca.autocast():
            with torch.no_grad():
                normdocx = [torch.norm(dx, dim=1).reshape(-1, 1) for dx in docs_torch]

        #Start training
        for epoch in range(nepoch):
            cbatches = np.random.permutation(len(docs_torch))  # we don't use the dataloader but rather permute the batch ids
            nAssign_t = torch.zeros(ntopics).to(device)  # number of assigned sentences to topic
            newvec_t = torch.zeros_like(vec_t).to(device)  # This will become the new topic vector v_t after the epoch
            #vecsToT=[[] for _ in range(ntopics)]
            for ib, i in enumerate(cbatches):  # go through all docs
                dx = docs_torch[i]  # current document
                with tca.autocast():
                    with torch.no_grad():
                        # Compute similarity of document, i.e., its sentences, and topic vectors
                        pts = torch.matmul(dx, vec_t)  #(sen x emb) (emb,top) # Dot product of sentence vectors in doc and topic vectors
                        #if cfg["cos"]:
                        nordx=normdocx[i]
                        normvect = torch.norm(vec_t, dim=0).reshape(1, -1)
                        pts=pts/(nordx*normvect+1e-7)

                        prod = pts * (ptd[:, i] + ioff)  # Compute p(s|t)*p(t|d) , (mean is not needed) #* torch.mean(pts[pts > 0]

                        vals, mind = torch.max(prod, dim=1)  # Compute most likely topic and take it as assignment
                        #if cfg["move"]:
                        if np.random.random() > 0.5 + 0.5 * (2*(epoch + 1) / nepoch):
                            prod[:, mind] = -1e5
                            vals2, mind2 = torch.max(prod, dim=1)
                            prod[:, mind] = vals
                            vals, mind = vals2, mind2
                        nor=torch.max(pts,dim=0)[0].reshape(1,-1)+1e-6
                        lprod = (pts/nor) #* ptd[:, i]#only for loss prob computatoin
                        lvals = lprod[:, mind]  # +lprod[:,mind2]

                        creal = pts * ptd[:, i]
                        creal, _ = torch.max(creal, dim=1)
                        rloss=(1-lalpha) * rloss + lalpha * torch.mean(creal).item() if epoch>0 or i>50 else 0.92 * rloss + 0.08 * torch.mean(creal).item() # update loss
                        mrloss = max(mrloss, rloss)


                        # Update p(t|d)
                        cntd = nn.functional.one_hot(mind, ntopics)
                        sntd = torch.sum(cntd, dim=0)
                        ptd[:, i] = sntd / (torch.sum(sntd)+1e-6)

                        # Update topic vectors
                        td = torch.transpose(dx, 0, 1)
                        prod = torch.matmul(td, cntd.float())
                        sntd = sntd * ptd[:, i]
                        prod = prod * ptd[:, i]
                        newvec_t = newvec_t + prod  # use only some sentence vectors -> could weight by ptd and do rolling update
                        nAssign_t += sntd

                        #if epoch>1: for itop,s in zip(mind,dx): vecsToT[itop].append(s)

                        # Other stuff:
                        assignSen_ptd[i] = mind.cpu().numpy()  # Remember assignment (only needed at the end)
                        prob_t[i]=vals.cpu().numpy()
                        # closs = (1-lalpha) * closs + lalpha * torch.mean(lvals).item() if epoch>0 or i>50 else 0.92 * closs + 0.08 * torch.mean(lvals).item() # update loss
                        # mloss=max(closs,mloss)
                        # if epoch==0 and i==5000:
                        #     rloss0=rloss
                        #     closs0=closs

            if self.verbose: print("  Epoch:",epoch)
            ioff=max(ioff / 2, alpha)
            with torch.no_grad():  # Update the topic vector
                newvec_t = newvec_t / (nAssign_t + 1e-7)
                #diff = torch.sqrt(torch.sum((newvec_t - vec_t) ** 2)) #early stopping: Compute loss difference
                vec_t = newvec_t
                #if diff < epsilon and epoch>4: break #early stopping: Stop if vectors don't change significantly anymore
        # cfg["rloss"] = closs
        # cfg["rmloss"] = mloss
        # cfg["rrloss"] = rloss
        # cfg["rmrloss"] = mrloss
        # cfg["rdelr"] = rloss-rloss0
        # cfg["rdel"] = closs-closs0
        return ptd.cpu().numpy(), vec_t, assignSen_ptd,prob_t


    """
    Lemmatize topics
    -----------
    Inputs
    lemDict...Dictionary of word -> lemma
    topics... list of list of words resembling top words of topics
    Returns
    lemmatized topics... list of list of words resembling top lemmatized words of topics
    """
    def lemTopics(self,topics, lemDict):
        nt=[]
        for t in topics:
            ct=[]
            for w in t:
                cw=w
                if cw in lemDict:
                    cw=lemDict[w]
                elif cw.lower() in lemDict:
                    cw=lemDict[cw.lower()]
                if not cw in ct:
                    ct.append(cw)
            nt.append(ct)
        return nt

    def get_topics(self,sendocs, assign_t,freqWeight=0.5):#we  for a word w and topic t, we compute (Full details are in paper)
        """
        Compute topics (list of words)
        -------------------
        Input
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentence;  ~ #docs x #sentences in doc
        ntopw ... number of words to return per topic (words with largest score are returned)
        freqWeight ... frequency weight in computation of word score for a topic. Typical values are from [0, 4], more narrowly, [0.25,1.5].  score(w|t) ~ (n(w,t)**freqWeight) * p(w|t) , where n(w,t) assignment of word w to topic t and p(w|t) is probability of word in a topic  (Details see paper).
                       A weight of 0 means that frequency is not relevant and a weight of 4 means that frequency is the key factor; That is, for a weight of 0, rather infrequent words that are only assigned to a topic get high scores, while for weight of 4 frequent word that might occur in many topics get high scores;
        -------------------
        Return
        topics ... list of lists of (words,score,frequency in topic) pairs; each list of (words,score,frequency in topic) pairs contains ntopw words (each with a score);
                   scores can be 0 if (i) no occurrence of the word is assigned to the topic; (ii) word are very infrequent, e.g., only occur a few times or just in a few documents; (iii) words are frequent and occur in many topics
        """
        #Preprocess corpus -> We need to tokenize documents into words, we also lower case and lemmatize
        # NOTE: Other methods use their own preprocessing, we also lemmatize the final topics of other methods
        dic = {}  # dictionary word to id
        nw = 2000  # number of initial words (this is grown)
        nfw = 0
        nd = len(sendocs)
        nt = self.ntopics
        occ_wt = np.zeros((nw, nt), dtype=np.int32)  # matrix with number of occ of word in topic (it is grown as needed)
        occ_wd = np.zeros((nw, nd), dtype=np.int32)  # matrix with number of occ of word in document
        occ_dt = np.zeros((nd, nt), dtype=np.int32)  # matrix with number of occ of topic per document
        max_occw = np.zeros(nw)  # max word in doc
        #from nltk.corpus import stopwords # Optional: Ignore stop words
        #stop_words = set(stopwords.words('english')) # Optional: Ignore stop words
        lemmatizer=WordNetLemmatizer()
        corpus = []
        lems={} #build map of words to lemmas
        for id in range(nd):  # go through all docs
            dass = assign_t[id]  # assignments of sentences of doc
            doc = sendocs[id]  # sentences in doc
            if self.verbose and len(dass) != len(doc):
                print("Failed not all sentences assigned", id, len(dass), len(doc)," NDoc", nd, len(sendocs),doc)
            cdoc = []
            c_occw = {}
            for it, s in enumerate(doc):  # go through all sentences
                toks = gut.tokenize(s, lower=True)
                toks = [ctok for ctok in toks if len(ctok) > 1]  # ignore single chars
                #toks = [ctok for ctok in toks if not ctok in stop_words]  # Optional: Ignore stop words
                for w in toks:
                    if not w in lems:
                        lems[w] = lemmatizer.lemmatize(w)
                toks = self.lemTopics([toks], lems)[0]
                ctopic = dass[it]
                for w in toks:  # go through all tokens
                    cdoc.append(w)
                    if not w in dic:  # add new word to dic
                        dic[w] = nfw
                        nfw += 1
                        if nfw > occ_wt.shape[0]:  # if found max number of words expand matrix
                            occ_wt = np.concatenate([occ_wt, np.zeros((nw, nt), dtype=np.int32)], axis=0)
                            occ_wd = np.concatenate([occ_wd, np.zeros((nw, nd), dtype=np.int32)], axis=0)
                            max_occw = np.concatenate([max_occw, np.zeros(nw, dtype=np.int32)], axis=0)
                    cind=dic[w]
                    if not cind in c_occw: c_occw[cind]=1
                    else: c_occw[cind] += 1
                    occ_wt[cind, ctopic] += 1  # add count of words in topic
                    occ_wd[cind, id] += 1
                    occ_dt[id, ctopic] += 1
            for ind in c_occw:
                max_occw[ind]=max(c_occw[ind],max_occw[ind])
            corpus.append(cdoc)

        denom = np.sum(occ_wt, axis=1)
        #avgdenom = denom/(np.sum(occ_wt > 0, axis=1) + 1e-8)
        denom=denom.reshape(-1, 1) +1e-8
        max_occw=max_occw.reshape(-1,1)
        std_occ=np.std(occ_wt,axis=1)
        #print(occ_wt.shape, std_occ.shape)
        #print(np.min(std_occ), np.max(std_occ), np.mean(std_occ), np.std(std_occ), std_occ[:10])
        std_occ=std_occ.reshape(-1,1)
        std_occ+=1
        aboveExp=np.clip((occ_wt - (denom / self.ntopics + max_occw)-std_occ),0,1e10)

        #scores_word_topic = (np.clip(aboveExp, 0, 1e10))**freqWeight * np.clip(((occ_wt-std_occ) / denom - max_occw / denom), 0, 1)
        scores_word_topic = (np.clip(aboveExp, 0, 1e10)) ** freqWeight * np.clip(occ_wt / denom -1.0/nt, 0, 1)
        #print(np.sort(max_occw)[:20],np.sort(max_occw)[-20:],np.mean(max_occw),np.std(max_occw),max_occw.shape,denom.shape)
        #print(np.clip(denom / self.ntopics,max_occw,1e10).shape)
        #scores_word_topic = (np.clip((occ_wt - np.clip(denom / self.ntopics,max_occw,1e10)),0,1e10)** freqWeight) * np.clip(occ_wt / denom - np.clip(max_occw/denom,1 / self.ntopics,1), 0,1) # Essentially, by look at how much more the probability is above "chance", the advantage is that otherwise we have strong dependency on the number of topics, i.e. , say we have ntopics=2 and the word "a" occurs 1000 times, by chance it occurs very frequent everywhere, byt for ntopics=200. by chance it occurs a factor 100 less # * np.sum(awt, axis=0,keepdims=True)/np.sum(awt)
        #scores_word_topic = (np.clip(occ_wt - 2*(denom / self.ntopics+ max_occw), 0, 1e10) ** freqWeight) * np.clip(occ_wt / denom - 2*max_occw / denom, 0, 1)  # Essentially, by look at how much more the probability is above "chance", the advantage is that otherwise we have strong dependency on the number of topics, i.e. , say we have ntopics=2 and the word "a" occurs 1000 times, by chance it occurs very frequent everywhere, byt for ntopics=200. by chance it occurs a factor 100 less # * np.sum(awt, axis=0,keepdims=True)/np.sum(awt)
        #scores_word_topic = (np.clip(occ_wt , 0, 1e10) ** freqWeight) * np.clip(occ_wt / denom , 0, 1)  # Essentially, by look at how much more the probability is above "chance", the advantage is that otherwise we have strong dependency on the number of topics, i.e. , say we have ntopics=2 and the word "a" occurs 1000 times, by chance it occurs very frequent everywhere, byt for ntopics=200. by chance it occurs a factor 100 less # * np.sum(awt, axis=0,keepdims=True)/np.sum(awt)

        #Get indexes of words with top scores
        sw=np.zeros((self.maxTopWords,nt),dtype=np.int32)
        swscores = np.zeros((self.maxTopWords, nt), dtype=np.float32)
        swfreq = np.zeros((self.maxTopWords, nt), dtype=np.float32)
        copy_scores_wt=np.copy(scores_word_topic)
        for i in range(self.maxTopWords):
            wind=np.argmax(copy_scores_wt,axis=0)
            mscore=np.max(copy_scores_wt,axis=0)
            sw[i]=wind
            swscores[i]=mscore
            swfreq[i]=occ_wt[wind,np.arange(nt)]
            copy_scores_wt[wind,np.arange(nt)]=-1 #set to maximum so that wont be taken again
        #print("done LO")
        #Get words per topic from indexes
        idic = {dic[w]: w for w in dic}
        topicsAsWordScoreList=[]
        for t in range(nt):
            ws=sw[:,t]
            scores=swscores[:,t]
            freqs=swfreq[:,t]
            #lws=[idic[w] for w in ws if w in idic]
            #lws = [(idic[w],s) for w,s in zip(ws,scores)]
            lws = [(idic[w], s,f) for w, s,f in zip(ws, scores,freqs) ]
            topicsAsWordScoreList.append(lws)
        #print("done")
        self.pt = np.mean(self.ptd, axis=1)
        #for p,t in zip(self.pt,topicsAsWordScoreList)[:10]: print(p,t)
        return topicsAsWordScoreList


    def segmenter(self,docs):
        """
        Segment docs into sentences
        Inputs: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        pid,docs,csen=docs
        seg = pysbd.Segmenter(language="en", clean=False)
        adocs=[]
        out=300
        #print(csen)
        p0,p1=1,csen #'nSen':(0,1) #combinations of sentences: (x,y) x=0 combine by length, merge i-1 and i if i or i-1 shorter than y tokens; x=1 combine static,i .e., combine y sentences
        amerges = []
        for id,d in enumerate(docs):
            segs = seg.segment(d)
            nmerges=0
            # if p0==0:
            #     ndoc = []
            #     j=0
            #     while j<len(segs)-2:
            #         if (segs[j]<p1 or segs[j+1]<p1):
            #             ndoc.append(segs[j]+segs[j+1]) #only merge 2 sentences
            #             j+=1
            #             nmerges+=1
            #         else:
            #             ndoc.append(segs[j])
            # elif p0==1:
            parts=self.split(segs, max(1,len(segs)//p1))
            ndoc=[" ".join(p) for p in list(parts)]
            nmerges+=len(segs)-len(ndoc)
            #elif p0==-1: ndoc=segs
            adocs.append(ndoc)
            amerges.append(nmerges)
            if self.verbose and id==out:
                print("   ProcessID:",pid," Segmented by process:",id,sum(amerges))
                out*=4
        return adocs,nmerges


    #Split list a into n parts
    def split(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    def getEncodedSentences(self,docs,device,cfg):
        """
        Segment docs into sentences
        Input: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        if self.verbose: print("Segmentation of docs into sentences; Total Docs: ",len(docs))
        #Segment documents into sentences
        nseg=min(8,multiprocessing.cpu_count()-1)
        px = list(self.split(docs, nseg))
        cpool = multiprocessing.Pool(nseg)
        docs_segmented = cpool.map(self.segmenter, zip(np.arange(len(px)),px,[self.nSenPerGroup]*len(px)))
        nmerges=sum([d[1] for d in docs_segmented],0)
        #cfg["nMerges"]=nmerges
        #print("tmerge",nmerges)
        docs_segmented = sum([d[0] for d in docs_segmented], [])
        #docs_segmented = sum(docs_segmented, [])
        filtered_docs=[]
        nFailed=0
        for i,d in enumerate(docs_segmented):
            if len(d)==0:
                if self.verbose and nFailed<5: print(" Removing empty sentence embedded doc; Original doc ID",i," Original length:",len(docs[i]), ("" if len(docs[i])==0 else "Content enclosed in '--' (e.g., it might contain only 'return characters'):  --"+docs[i]+"--"))
                nFailed+=1
            else:
                filtered_docs.append(d)
        if nFailed>0: print(" Number of empty sentenced embedded docs:",nFailed, ("  (Use verbose option to see first 5 docs)" if not self.verbose else ""))
        if self.verbose: print("Embed sentences using a sentence embedder; Total Docs: ",len(filtered_docs))
        #get Sentence Embeddings -> this can be parallelized
        model = SentenceTransformer(self.ftname, device=device)
        model.eval()
        docs_sentenceEncoded = []
        out = 300
        for id, d in enumerate(filtered_docs):
            with torch.no_grad():
                with tca.autocast():
                    segs = model.encode(d)
            docs_sentenceEncoded.append(segs)
            if self.verbose and id == out:
                print("  Encoded Docs:", id)
                out *= 4
        return filtered_docs,docs_sentenceEncoded


    def fit_transform(self, docs, nTopics=50, alpha=None, nEpochs=10, loadAndStoreInFolder=None, verbose=True,seed=4711,freqWeight=0.5,cfg=None):
        """
        Compute topic_model given raw text docs, optional: stores partial sentence embeddings for recomputation with other parameters
        --------------------
        Input:
        docs...documents
        alpha ... parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document), default is 1/sqrt(nTopics);  0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha (even if no sentence of document belongs to topic t)
        loadAndStoreInFolder...path to a folder, where docs (split into sentences) and sentence embeddings of docs are stored; if they exist, these are loaded and used for topic modeling ; this speeds up computation of different models on same corpus
        verbose...Print state of computation during topic modeling process
        seed...Random seed to be used; Use None, if you don't want to set a seed
        freqWeight ... frequency weight in computation of word score for a topic. Typical values are from [0, 4], more narrowly, [0.25,1.5].  score(w|t) ~ (n(w,t)**freqWeight) * p(w|t) , where n(w,t) assignment of word w to topic t and p(w|t) is probability of word in a topic  (Details see paper).
                       A weight of 0 means that frequency is not relevant and a weight of 4 means that frequency is the key factor; That is, for a weight of 0, rather infrequent words that are only assigned to a topic get high scores, while for weight of 4 frequent word that might occur in many topics get high scores;
        Returns:
        topics...list of topics, each topic contains a list of "ntopwords" words sorted in descending order by relevance
        probs...topic probabilities
        """
        if not seed is None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        if alpha is None: alpha=1/np.sqrt(nTopics)
        if self.device=='auto':
            device='cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose=verbose
        # get encoded Sentences
        if not loadAndStoreInFolder is None:
            topicDatName=loadAndStoreInFolder + "/topicDat.pic"
        if loadAndStoreInFolder is None or not os.path.exists(topicDatName):
            start = time.time()
            self.sendocs, embdocs = self.getEncodedSentences(docs, device,cfg)
            self.timePrePro= time.time() - start
            if not loadAndStoreInFolder is None:
                os.makedirs(loadAndStoreInFolder, exist_ok=True)
                with open(topicDatName, "wb") as f:
                    pickle.dump([self.sendocs, embdocs], f)
        else:
            if self.verbose: print("Loading docs split into sentences and embedded sentences for faster topic modeling")
            with open(topicDatName, "rb") as f:
                 self.sendocs, embdocs = pickle.load(f)
            self.timePrePro =-1


        self.ntopics=nTopics
        # Topic modeling and word-topic extraction
        start = time.time()
        self.ptd, self.vec_t, self.assignSen_ptd,self.prob_t=self.computeTopicModel_SenClu(embdocs, device, nTopics, nepoch=nEpochs, alpha=alpha,cfg=cfg)
        self.topicsAsWordScoreList = self.get_topics(self.sendocs, self.assignSen_ptd,freqWeight=freqWeight)
        self.timeTopic = time.time() - start
        return self.getTopics(cfg=cfg),self.pt

    def getTopics(self,wordsPerTopic=10,cfg=None):
        #print("Alltop",self.topicsAsWordScoreList)
        topics=[[(w[0] if w[1]>0 else "") for w in t[:wordsPerTopic] ] for t in self.topicsAsWordScoreList]
        #print(topics)
        finaltopics=[]
        removedTopics,removedWords=0,0
        for t in topics:
            ctopic= [w for w in t if len(w)>0]
            if len(ctopic)>0:
                finaltopics.append(ctopic)
            else:
                finaltopics.append([""])
                removedTopics+=1
            removedWords+=wordsPerTopic-len(ctopic)
        if removedWords>0: print("Rare words or words common in all topics or words with many occ in other topics (but not in this) got removed: ",removedWords," words;   Complete topics removed:",removedTopics)
        # cfg["rremW"] = removedWords
        # cfg["rremT"] = removedTopics
        return finaltopics


    def getTopicsWithScores(self): return self.topicsAsWordScoreList

    def getTopicDocumentDistribution(self): return self.ptd

    def getTopicDistribution(self): return self.pt

    def getTopicVecs(self): return self.vec_t.cpu().numpy() #get embedding vectors of all topics; Only used for evaluation

    def getSenAssign(self): return self.assignSen_ptd #get assignments of sentences to topics; Only used for evaluation


    def getTopDocsPerTopic(self,ntop=20):
        sortDocs=np.argsort(self.ptd,axis=1) #ascending order!
        topDocs=[]
        for i in range(self.ntopics):
            topd=sortDocs[i][-ntop:]
            topd=topd[::-1] #descending order
            scores=self.ptd[i][topd]
            csen=[self.sendocs[itop] for itop in topd]
            topDocs.append(list(zip(csen,scores,topd)))
        return topDocs #return document ID, doc score and doc


    def getTopSentencesPerTopic(self, ntop=20):
        perTop=[[] for _ in range(self.ntopics)]
        for i in range(len(self.sendocs)):
            vals=self.prob_t[i]
            tops=self.assignSen_ptd[i]
            for j in range(len(vals)):
                perTop[tops[j]].append((vals[j],i,j))
        for t in range(self.ntopics):
            s=sorted(perTop[t],reverse=True)
            ct=[]
            for ctop in range(ntop):
                if len(s)<=ctop: break
                v,i,j=s[ctop]
                ct.append((self.sendocs[i][j],v))
            perTop[t]=ct
        return perTop



    def saveOutputs(self, folder="Bert-SenClu", topWordPerTopic=10, topSenPerTopic=10, topDocsPerTopic=10, maxSenPerDoc=50, addTopicMarkup=True,createVisual=True):
        """
        Produce csv and visualization files and store in given folder; The csv can also be used for exploration and visualization
        A topic contains a probability, top words (each with probability), top sentences (each with score), top documents for that topic (full docs, only topic sentences, topic sentences with context); each doch has a score and an ID of the original document in the dataset fed into the topic model)
        For top documents, three representations are output:
        - "FullDoc" The full document up to specified maximum number of sentences maxSenPerDoc
        - "ContextDoc"  The document always contains the first 5 and the last 2 sentences and otherwise only sentences belonging to the topic (plus 1 sentence before/after to give context)
        - "TopicOnlyDoc" Keep only sentences belonging to the document
        The csv file contains 1 row for each topic and each column corresponds to a single item, e.g., a topic word, a topic probability, a top sentence etc.
        -------------------------------------------------
        Input
        fileName ... name of csv file
        topSenPerTopic=15... number of top sentences per topic
        topDocsPerTopic=20 ... number of top documents per topic
        maxSenPerDoc=100 ... maximum number of displayed sentences per document
        addTopicMarkup=True ... for a top document of a topic, highlight sentences belonging to topic
        createVisual=True ... create and store visualization files
        Returns:
        None (stores files in given folder)
        """
        if self.verbose: print("Creating and storing Outputs")
        #get data
        senPerTop=self.getTopSentencesPerTopic(topSenPerTopic)
        docPerTop=self.getTopDocsPerTopic(topDocsPerTopic)

        #Compute  CSV
        header = ["Topic_ID", "Topic_Prob"]
        r=lambda x: str(np.round(x,4))
        if topWordPerTopic>self.maxTopWords:
            print("Exceeded max words per Topic. Fixing to allowed maximum of ",self.maxTopWords)
            topWordPerTopic=self.maxTopWords
        headW=sum([["Word_"+str(i),"Prob_"+str(i)] for i in range(topWordPerTopic)],[])
        headS = sum([["Sentence_" + str(i), "Prob_" + str(i)] for i in range(topSenPerTopic)], [])
        headD = sum([["FullDoc_" + str(i),"ContextDoc_"+ str(i),"TopicOnlyDoc_"+str(i),"Prob_" + str(i),"DocID_"+str(i)] for i in range(topDocsPerTopic)], [])
        rows = []

        def getFixedLenList(li,nEle):
            conLi = [[w[0], r(w[1])] for w in li[:nEle]]
            conLi=sum(conLi,[])
            return conLi

        topicColor='#D84141'
        for i in range(self.ntopics):
            topdat = [str(i),r(self.pt[i])]
            topW=getFixedLenList(self.topicsAsWordScoreList[i], topWordPerTopic)
            topS=getFixedLenList([(("<span style='color:"+topicColor+"'>"  + s[0]  + "</span>") if addTopicMarkup else s[0],s[1]) for s in senPerTop[i]], topSenPerTopic)

            topD=docPerTop[i]
            conDoc=[]
            #Add Docs - keep only sentences of the topic i and for each topic sentence one before and after as well as first and last sentences

            for d,dsc,did in topD:
                topSen=(np.array(self.assignSen_ptd[did]) == i).astype(np.int32)
                senToKeep = np.copy(topSen)
                senToKeep[:5]=1 #keep first and last sentences
                senToKeep[-2:] = 1
                senToKeep[:-1]+=topSen[1:] #for each topic sentence add one before and after of a different topic
                senToKeep[1:] += topSen[:-1]
                consens,topsens,alls = [],[],[]
                thres = np.median(self.prob_t[did][topSen>0]) # highlight top 50% of a sentence more
                skipSen=0
                for inds,s in enumerate(d):
                    if topSen[inds]: #Topic sentence?
                        if addTopicMarkup:  # add markup if topic sentence
                            bold = self.prob_t[did][inds] > thres
                            s = "<span style='color:"+topicColor+"'>" + ("<b>" if bold else "") + s + ("</b>" if bold else "") + "</span>" #redish color
                        if len(topsens) < maxSenPerDoc: topsens.append(s)
                    elif addTopicMarkup:  # add markup if topic sentence
                        s = "<span style='color:#7D88C7'>"+ s + "</span>"  # blue color
                    if len(alls) < maxSenPerDoc: alls.append(s)
                    if senToKeep[inds]:
                        if skipSen:  # if left out a sentence add dots
                            s = " ...>>" + str(skipSen) + "... " + s
                        if len(consens) < maxSenPerDoc: consens.append(s)
                        skipSen=0
                    else:
                        skipSen+=1
                conDoc.append([" ".join(alls), " ".join(consens), " ".join(topsens if len(topsens) else [" "]),r(dsc),str(did)])
            conDoc = sum(conDoc, [])
            topdat+=topW+topS+conDoc
            rows.append(topdat)

        df=rows
        df = pd.DataFrame(df, columns = header+headW+headS+headD)
        colW = [c for c in df.columns if c.startswith("Word_")]
        df["TopicShort"] = df.apply(lambda r: "_".join([str(r["Topic_ID"])] + list(r[colW].values[:5]))[:60], axis=1)
        os.makedirs(folder, exist_ok=True)
        df.to_csv(folder+"/topic_all_info.csv")
        nvec=np.transpose(self.vec_t.cpu().numpy())


        #Create visualization of topics
        if createVisual:
            from bertsenclu import visUtils
            if self.verbose: print("Creating Visualizations")
            visUtils.hierarchy(folder, nvec, df["TopicShort"].values)
            visUtils.tsne(self, nvec, folder, df["TopicShort"].values)



