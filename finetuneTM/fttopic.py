import sys
import numpy as np
import mkl
import pysbd #Sentence segmenter https://github.com/nipunsadvilkar/pySBD #pip install pysbd
import random
from datasets import Dataset
from sentence_transformers import (SentenceTransformer,SentenceTransformerTrainer,)
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
dtype=np.float32
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
mkl.set_num_threads(4)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #needed for huggingface transformer, otherwise get warning "The current process just got forked, Disabling parallelism to avoid deadlocks.. To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)"
import multiprocessing


def getFineTuningTrainingData(senDocs, model, nneg=2, ftri=0.24, fpos=0.08, fneg=0):
    """
    Constructs training data for fine-tuning a language model for topic analysis.
    This function generates triplets of sentences (anchor, positive, negative) from a list of sentence documents.
    Positive sentences are close to the anchor in the same document, while negative sentences are randomly chosen
    from different documents.

    Args:
    senDocs (list of list of str): List of documents with each document being a list of sentences.
    model (SentenceTransformer): Pre-trained sentence transformer model to encode sentences.
    nneg (int, optional): Number of negative samples per positive sample. Defaults to 2.
    ftri (float, optional): Fraction of triplets to filter based on distance criteria. Defaults to 0.24.
    fpos (float, optional): Fraction of positive pairs to filter out based on distance. Defaults to 0.08.
    fneg (float, optional): Fraction of negative pairs to keep if they are close to anchor. Defaults to 0.

    Returns:
    Dataset: A dataset containing 'anchor', 'positive', and 'negative' sentences for training.
    """
    num_docs = len(senDocs)
    if num_docs<2:
        print("Need at least two documents but got only",num_docs)
        return None
    lanc,lpos,lneg=[],[],[]
    for doc_index in range(num_docs):
        doc_sentences = senDocs[doc_index]
        if len(doc_sentences)<2: continue
        for sent_index in range(len(doc_sentences)):
            anchor = doc_sentences[sent_index]
            if sent_index == 0: inds=[sent_index + 1]
            elif sent_index == len(doc_sentences) - 1: inds=[sent_index - 1]
            else: inds=[sent_index - 1,sent_index + 1]
            # Determine negative sentence from another document
            for si in inds:
              positive = doc_sentences[si]
              for j in range(nneg):
                other_doc_index = random.choice([i for i in range(num_docs) if i != doc_index])
                other_doc_sentences = senDocs[other_doc_index]
                negative = random.choice(other_doc_sentences)
                lanc.append(anchor)
                lpos.append(positive)
                lneg.append(negative)

    def batch_encode(lanc, batch_size=512):
        encoded_batches = []
        for i in range(0, len(lanc), batch_size):
            batch = lanc[i:i + batch_size]
            encoded_batch = model.encode(batch)
            encoded_batches.extend(encoded_batch)
        return np.array(encoded_batches)

    if fpos+ ftri+fneg>0:
        #print("Filtering",tofilter,split)
        eanc = batch_encode(lanc)
        epos = batch_encode(lpos)
        eneg = batch_encode(lneg)

        #Remove those with large !eanc-eneg! - !eanc-epos!
        dist_pos = np.linalg.norm(eanc - epos, axis=1)
        dist_neg = np.linalg.norm(eanc - eneg, axis=1)
        diff = dist_neg -dist_pos
        threshold = np.percentile(diff, ftri)
        top20 = np.where(diff >= threshold)[0]

        lanc, lpos, lneg = np.array(lanc)[top20], np.array(lpos)[top20], np.array(lneg)[top20]
        # Remove those with min -!eanc-epos!
        dist_pos = -dist_pos[top20]
        threshold = np.percentile(dist_pos, ftri)
        top20 = np.where(dist_pos >= threshold)[0]
        lanc, lpos, lneg = np.array(lanc)[top20], np.array(lpos)[top20], np.array(lneg)[top20]

        # Remove those with min !eanc-eneg!
        dist_neg = dist_neg[top20]
        threshold = np.percentile(dist_neg,fneg)
        top20 = np.where(dist_neg >= threshold)[0]
        #removed = np.where(dist_neg < threshold)[0]        #rem3 = dist_neg[removed]        #l0 = len(lanc) #mima=lambda x:np.round([np.mean(x),np.min(x),np.max(x)],4) if len(x)>0 else "" #print("Filtered data for FT",l0,len(lanc),tofilter," Removed diffs",mima(rem)," diffs",mima(diff)," Removed large eanc-epos",mima(rem2)," diffs",mima(dist_pos)," Removed small eanc-eneg",mima(rem3)," diffs",mima(dist_neg),split)
        lanc,lpos,lneg=np.array(lanc)[top20].tolist(),np.array(lpos)[top20].tolist(),np.array(lneg)[top20].tolist()
    return Dataset.from_dict({'anchor': lanc, 'positive': lpos, 'negative': lneg})


def split( a, n):
    """
    Divides a list into n approximately equal parts.

    Args:
    a (list): The list to split.
    n (int): The number of equal parts.

    Returns:
    generator: A generator yielding each part as a sublist.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def getSentences( docs,nSenPerGroup=3):
    """
    Processes a list of documents into a list of sentences grouped by proximity within the document.
    It uses multiprocessing for faster execution and segments documents into sentences based on the specified
    number of sentences per group.

    Args:
    docs (list of str): List of documents, each document being a string.
    nSenPerGroup (int, optional): Number of sentences per group for segmentation. Defaults to 3.

    Returns:
    list of list of str: A list of documents where each document is a list of sentence groups.
    """
    print("Segmentation of docs into sentences; Total Docs: ", len(docs))
    # Segment documents into sentences
    nseg = min(10, multiprocessing.cpu_count() - 1)
    px = list(split(docs, nseg))
    cpool = multiprocessing.Pool(nseg)
    docs_segmented = cpool.map(segmenter, zip(np.arange(len(px)), px, [nSenPerGroup] * len(px)))

    docs_segmented = sum([d[0] for d in docs_segmented], [])
    filtered_docs = []
    nFailed = 0
    for i, d in enumerate(docs_segmented):
        if len(d) == 0:
            #if self.verbose and nFailed < 5: print(" Removing empty sentence embedded doc; Original doc ID", i, " Original length:", len(docs[i]), ("" if len(docs[i]) == 0 else "Content enclosed in '--' (e.g., it might contain only 'return characters'):  --" + docs[i] + "--"))
            nFailed += 1
        else:
            filtered_docs.append(d)
    #if nFailed > 0: print(" Number of empty sentenced embedded docs:", nFailed, ("  (Use verbose option to see first 5 docs)" if not self.verbose else ""))
    print("Embed sentences using a sentence embedder; Total Docs: ", len(filtered_docs))
    return filtered_docs

def segmenter(docs):
        """
        Segment docs into sentences
        Inputs: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        pid,docs,csen=docs
        seg = pysbd.Segmenter(language="en", clean=False)
        adocs=[]
        out=300
        p0,p1=1,csen
        amerges = []
        for id,d in enumerate(docs):
            segs = seg.segment(d)
            nmerges=0
            parts=split(segs, max(1,len(segs)//p1))
            ndoc=[" ".join(p) for p in list(parts)]
            nmerges+=len(segs)-len(ndoc)
            adocs.append(ndoc)
            amerges.append(nmerges)
            if id==out:
                print("   ProcessID:",pid," Segmented by process:",id,sum(amerges))
                out*=4
        return adocs,nmerges


def getFTTopic(documents,modelToTune="all-MiniLM-L6-v2",outputName="tunedEncoder",margin=0.16,ftri=0.24,fpos=0.08,nneg=2,ep=4,trainingFrac=1,device = "cuda"):
    """
    Fine-tunes a transformer model for topic extraction from a collection of documents.
    This function configures and initiates the training process based on the provided documents and model,
    aiming to improve the model's ability to differentiate between the topics of the documents.

    Args:
    documents (list of str): List of documents to be used for training the model.
    modelToTune (str, optional): Identifier for the pre-trained model to be fine-tuned. Defaults to "all-MiniLM-L6-v2".
    outputName (str, optional): The filename to save the fine-tuned model. Defaults to "tunedEncoder".
    margin (float, optional): Margin for triplet loss during training. Defaults to 0.16.
    ftri (float, optional): Fraction of triplets to filter based on distance criteria. Defaults to 0.24.
    fpos (float, optional): Fraction of positive pairs to filter out based on distance. Defaults to 0.08.
    nneg (int, optional): Number of negative samples per positive sample. Defaults to 2.
    ep (int, optional): Number of epochs to train the model. Defaults to 4.
    trainingFrac (float, optional): Fraction of the provided documents to use for training. Defaults to 1.
    device (str, optional): The device to perform training on (e.g., 'cuda' or 'cpu'). Defaults to "cuda".

    Returns:
    None: The function saves the fine-tuned model to the specified output file and does not return any value.
    """
    model = SentenceTransformer(modelToTune, device=device)
    senDocs = getSentences(documents)
    if trainingFrac<1: random.shuffle(senDocs)
    split_index = int(trainingFrac * len(senDocs))  # Calculate split point for 90% train, 10% test
    trtriplets = getFineTuningTrainingData(senDocs[:split_index], model, nneg=nneg, ftri=ftri, fpos=fpos, fneg=0)
    evtriplets = getFineTuningTrainingData(senDocs[split_index:], model, nneg=nneg, ftri=ftri, fpos=fpos, fneg=0)

    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    args = SentenceTransformerTrainingArguments(
        output_dir="senOut",
        num_train_epochs=ep,
        per_device_train_batch_size=32, #should adjust based on GPU memory, e.g. for H100 with 90 GB recommend 128
        per_device_eval_batch_size=64, #should adjust based on GPU memory, e.g. for H100 with 90 GB can easily use 512
        warmup_ratio=0.1,
        fp16=False,  # Set to False if your GPU can't handle FP16
        bf16=True,  # Set to True if your GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        evaluation_strategy="steps",
        eval_steps=200000,
        save_strategy="steps",
        save_steps=1000000,
        save_total_limit=1,
        logging_steps=1000,
    )
    if trainingFrac<1:
        dev_evaluator = TripletEvaluator(
            anchors=evtriplets["anchor"],
            positives=evtriplets["positive"],
            negatives=evtriplets["negative"],
            name="all-genDat-dev",)
        print("Eval before",dev_evaluator(model))

    loss=TripletLoss(model,triplet_margin=margin) #5 is much worse than 0.33, maybe lower than 0.33 is even better
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=trtriplets,
        eval_dataset=evtriplets,
        loss=loss,
        evaluator=dev_evaluator if trainingFrac<1 else None,
    )
    trainer.train()
    if trainingFrac < 1: print("Eval after", dev_evaluator(model))

    model.save_pretrained(outputName)
    print("Model fine-tuned and saved under ",outputName)
