import os
import csv
import nltk
import json
import torch
import pickle
import random
import logging
import numpy as np
from ast import literal_eval
from copy import deepcopy
from tqdm.auto import tqdm
from src.utils.config import config
import torch.utils.data as data
from collections import defaultdict
from src.utils.common import save_config
from src.models.common import gen_embeddings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.translate.bleu_score import corpus_bleu
from src.utils.constants import ESC_EMO_MAP as esc_emo_map
from src.utils.constants import ESC_STRATEGY_MAP as esc_strategy_map
from src.utils.constants import ED_DATA_FILES
from src.utils.constants import ED_EMO_MAP as ed_emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from src.utils.constants import REMOVE_RELATIONS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")
wnl = WordNetLemmatizer()

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None

def _norm(x):
    return ' '.join(x.strip().split())

def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

def clear_commonsense(cs_res):
    # relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
    original_res = deepcopy(cs_res)
    clear_res = defaultdict(list)
    selected_res = []
    # pointer
    pointer_idx = dict([(rel,0) for rel in relations])
    weights = [1,0,0,0]
    done = [False] * len(relations)
    for i in range(10):
        for j, rel in enumerate(relations):
            if pointer_idx[rel] < len(original_res[rel]):
                commonsense = original_res[rel][pointer_idx[rel]]
                if len(selected_res) == 0:
                    selected_res.append(commonsense)
                    clear_res[rel].append(commonsense)
                    pointer_idx[rel] += 1
                else:
                    bleu_matrix = []
                    for cs in selected_res:
                        if cs[0] == "to":
                            cmp_cs = cs[1:]
                        else:
                            cmp_cs = cs
                        if commonsense[0] == "to":
                            cmp_commonsense = commonsense[1:]
                        else:
                            cmp_commonsense = commonsense
                        bleu_matrix.append(corpus_bleu([[cmp_cs]], [cmp_commonsense],weights=weights)) # don't consider "to"
                    if max(bleu_matrix)<=0.5:
                        selected_res.append(commonsense)
                        clear_res[rel].append(commonsense)
                    pointer_idx[rel] += 1
            else:
                done[j] = True
        if all(done):
            break
    return_res = dict()
    for rel in relations:
        return_res[rel] = clear_res[rel][:5]
        if len(return_res[rel]) < 5:
            return_res[rel] = return_res[rel] + ["none"]*(5-len(return_res[rel]))
    return return_res

def get_esconv_commonsense(comet, item):
    cs_list = defaultdict(list)
    input_event = " ".join(item)
    cs_list["uttr"] = item
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(_norm(item.replace(".",""))) for item in cs_res if item.strip() != "none" and item.strip() != "."]
        cs_list[rel] = cs_res
    # clear_cs = clear_commonsense(cs_list)
    # for rel in relations:
    #     cs_list[rel] = deepcopy(clear_cs[rel])
    return cs_list

def get_esconv_split_commonsense(comet, item):
    input_events = [process_sent(_norm(event)) for event in " ".join(item).split(".") if len(process_sent(_norm(event)))>0]
    uttr_cs_list = []
    for event in input_events:
        uttr_cs_list.append(get_esconv_commonsense(comet, event))
    assert len(input_events) == len(uttr_cs_list)
    return uttr_cs_list

def encode_esconv(data, vocab):
    """
    return data_dict = {
        "context_role": [],
        "context": [[[c1],[c2],[c3]]],
        "strategy": [],
        "context_strategy_seqs": [[]]
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [["uttr":[],"rel":[],"rel":[]]],  # commonsense
        "utt_cs_split": [["uttr_split":[[],[],[]],"cs":[["rel":[],"rel":[]],[],[]]]] # uttr is split by ".", futher obtaion the commonsense of each sentence
    }
    """
    from src.utils.comet import Comet
    data_dict = {
        "context_role": [],
        "situation": [],
        "context": [],
        "context_strategy_seqs": [],
        "strategy": [],
        "target": [],
        "emotion": [],
        "emotion_context": [],
        "utt_cs": [],
        "utt_cs_split": [],
        "situation_cs": []
    }
    comet = Comet(config.comet_file, config.device)
    for line in tqdm(data):
        context_role = line["context_role"] # list
        context_strategy_seqs = line["context_strategy_seqs"]
        emotion_type = line["emotion_type"] # str
        situation = process_sent(_norm(line["situation"])) # list
        vocab.index_words(situation)
        target = process_sent(_norm(line["target"]))
        vocab.index_words(target)
        strategy = line["strategy"]
        vocab.index_words(process_sent(_norm(line["strategy"])))
        context_list = []
        emotion_list = []
        for uttr in line["context"]:
            item = process_sent(_norm(uttr))
            context_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ) and w[0] not in emotion_list:
                    emotion_list.append(w[0])
        last_seeker_idx = max(index for index, role in enumerate(context_role) if role == "seeker")
        utt_cs = get_esconv_commonsense(comet, context_list[last_seeker_idx])
        utt_cs_split = get_esconv_split_commonsense(comet, context_list[last_seeker_idx])
        situation_cs = get_esconv_commonsense(comet, situation)
        data_dict["context"].append(context_list)
        data_dict["context_role"].append(context_role)
        data_dict["context_strategy_seqs"].append(context_strategy_seqs)
        data_dict["situation"].append(situation)
        data_dict["emotion"].append(emotion_type)
        data_dict["emotion_context"].append(emotion_list)
        data_dict["strategy"].append(strategy)
        data_dict["target"].append(target)
        data_dict["utt_cs"].append(utt_cs)
        data_dict["utt_cs_split"].append(utt_cs_split)
        data_dict["situation_cs"].append(situation_cs)
    assert len(data_dict["context"])==len(data_dict["context_role"])==len(data_dict["situation"])==len(data_dict["emotion"])==len(data_dict["emotion_context"])==len(data_dict["target"])==len(data_dict["utt_cs"])==len(data_dict["utt_cs_split"])==len(data_dict["situation_cs"])==len(data_dict["context_strategy_seqs"])==len(data_dict["strategy"])
    return data_dict

def read_esconv(data, vocab):
    dev_file = f"{config.data_dir}/{config.dataset}/dev_data.json"
    test_file = f"{config.data_dir}/{config.dataset}/test_data.json"
    train_file = f"{config.data_dir}/{config.dataset}/train_data.json"
    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        # Split Train/dev/Test
        print("Split Train/Dev/Test......")
        total_data = []
        for session in tqdm(data):
            experience_type = session["experience_type"]
            emotion_type = session["emotion_type"]
            problem_type = session["problem_type"]
            situation = session["situation"]
            dialog = session["dialog"]
            context = []
            context_role = []
            context_strategy_seqs = []
            for info in dialog:
                speaker = info["speaker"]
                if speaker == "seeker":
                    content = info["content"]
                    context.append(content)
                    context_role.append(speaker)
                    context_strategy_seqs.append("none")
                elif speaker == "supporter":
                    assert "strategy" in info["annotation"]
                    strategy = info["annotation"]["strategy"]
                    target = info["content"]
                    if "seeker" in context_role[-config.max_num_dialog:]:
                        save_data = {
                            "experience_type": experience_type,
                            "emotion_type": emotion_type,
                            "problem_type": problem_type,
                            "situation": situation,
                            "strategy": strategy,
                            "context_strategy_seqs": deepcopy(context_strategy_seqs[-config.max_num_dialog:]),
                            "context_role": deepcopy(context_role[-config.max_num_dialog:]),
                            "context": deepcopy(context[-config.max_num_dialog:]),
                            "target": target
                        }
                        total_data.append(save_data)
                    context.append(target)
                    context_role.append(speaker)
                    context_strategy_seqs.append(strategy)
                else:
                    raise ValueError(speaker + " Wrong!")
        random.seed(config.split_data_seed)
        random.shuffle(total_data)
        dev_size = int(0.1 * len(total_data))
        test_size = int(0.1 * len(total_data))
        dev_data = total_data[:dev_size]
        test_data = total_data[dev_size: dev_size+test_size]
        train_data = total_data[dev_size+test_size:]
        with open(train_file,"w",encoding="utf-8") as f:
            json.dump(train_data, f, indent=2)
        with open(dev_file,"w",encoding="utf-8") as f:
            json.dump(dev_data, f, indent=2)
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2)
        print("Total Data: ", len(total_data))
        print("Train Data: ", len(train_data))
        print("Dev Data: ", len(dev_data))
        print("Test Data: ", len(test_data))
        print("Split Complete!")
    else:
        train_data = json.load(open(train_file,"r",encoding="utf-8"))
        dev_data = json.load(open(dev_file,"r",encoding="utf-8"))
        test_data = json.load(open(test_file, "r", encoding="utf-8"))
    
    data_train = encode_esconv(train_data, vocab)
    data_dev = encode_esconv(dev_data, vocab)
    data_test = encode_esconv(test_data, vocab)
    
    return data_train, data_dev, data_test, vocab
   
def read_ed(vocab, data):
    data_list = []
    context = data[0]
    target = data[1]
    emotion = data[2]
    situation = data[3]
    for i in range(len(context)):
        context_role = []
        for j in range(len(context[i])):
            if j % 2==0:
                context_role.append("seeker")
            else:
                context_role.append("support")
        save_data = {
            "experience_type": None,
            "problem_type": None,
            "strategy": "Other",
            "context_strategy_seqs": "Other",
            "emotion_type": emotion[i],
            "situation": situation[i],
            "context": context[i],
            "context_role": context_role,
            "target": target[i]
        }
        data_list.append(save_data)
    data_return = encode_esconv(data_list, vocab)
    return data_return
 
def read_files(dataset=None, data_file=None, vocab=None):
    if dataset == "ESConv":
        total_data = json.load(open(data_file, encoding="utf-8"))
        print("Num of Original Dialog Session: ", len(total_data))
        data_train, data_dev, data_test, vocab = read_esconv(total_data, vocab)
    elif dataset == "ED":
        files = ED_DATA_FILES(config.data_dir)
        train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
        dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
        test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

        data_train = read_ed(vocab, train_files)
        data_dev = read_ed(vocab, dev_files)
        data_test = read_ed(vocab, test_files)
    else:
        raise ValueError(dataset + "does not exist!")
    return data_train, data_dev, data_test, vocab

def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468

def get_concept_dict(data_dir, dataset, vocab):
    VAD = json.load(open(f"{data_dir}/VAD.json", "r", encoding="utf-8"))
    CN = csv.reader(open(f"{data_dir}/assertions.csv", "r", encoding="utf-8"))
    concept_file = open(f"{data_dir}/{dataset}/ConceptNet.json", "w", encoding="utf-8")
    rd = open(f"{data_dir}/{dataset}/relation.json", "w", encoding="utf-8")
    embeddings = gen_embeddings(vocab)
    word2index = vocab.word2index
    concept_dict = {}
    relation_dict = {}
    for i, row in enumerate(CN):
        if i%1000000 == 0:
            print("Processed {} rows".format(i))
        items = "".join(row).split("\t")
        c1_lang = items[2].split("/")[2]
        c2_lang = items[2].split("/")[2]
        if c1_lang == "en" and c2_lang == "en":
            if len(items) != 5:
                print("concept error!")
            relation = items[1].split("/")[2]
            c1 = items[2].split("/")[3]
            c2 = items[3].split("/")[3]
            c1 = wnl.lemmatize(c1)
            c2 = wnl.lemmatize(c2)
            weight = literal_eval("{" + row[-1].strip())["weight"]

            if weight < 1.0:  # filter tuples where confidence score is smaller than 1.0
                continue
            if c1 in word2index and c2 in word2index and c1 != c2 and c1.isalpha() and c2.isalpha():
                if relation not in word2index:
                    if relation in relation_dict:
                        relation_dict[relation] += 1
                    else:
                        relation_dict[relation] = 0
                c1_vector = torch.Tensor(embeddings[word2index[c1]])
                c2_vector = torch.Tensor(embeddings[word2index[c2]])
                c1_c2_sim = torch.cosine_similarity(c1_vector, c2_vector, dim=0).item()

                v1, a1, d1 = VAD[c1] if c1 in VAD else [0.5, 0.0, 0.5]
                v2, a2, d2 = VAD[c2] if c2 in VAD else [0.5, 0.0, 0.5]
                emotion_gap = 1-(abs(v1-v2) + abs(a1-a2))/2
                # <c1 relation c2>
                if c2 not in stop_words:
                    c2_vad = emotion_intensity(VAD, c2) if c2 in VAD else 0.0
                    # score = c2_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c2_vad + emotion_gap
                    if c1 in concept_dict:
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c1] = {}
                        concept_dict[c1][c2] = [relation, c2_vad, c1_c2_sim, weight, emotion_gap, score]
                # reverse relation  <c2 relation c1>
                if c1 not in stop_words:
                    c1_vad = emotion_intensity(VAD, c1) if c1 in VAD else 0.0
                    # score = c1_vad + c1_c2_sim + (weight - 1) / (10.0 - 1.0) + emotion_gap
                    score = c1_vad + emotion_gap
                    if c2 in concept_dict:
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]
                    else:
                        concept_dict[c2] = {}
                        concept_dict[c2][c1] = [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]

    print("concept num: ", len(concept_dict))
    json.dump(concept_dict, concept_file)

    relation_dict = sorted(relation_dict.items(), key=lambda x: x[1], reverse=True)
    json.dump(relation_dict, rd)

def rank_concept_dict(data_dir, dataset):
    concept_dict = json.load(open(f"{data_dir}/{dataset}/ConceptNet.json", "r", encoding="utf-8"))
    rank_concept_file = open(f"{data_dir}/{dataset}/ConceptNet_VAD_dict.json", "w", encoding="utf-8")
    rank_concept = {}
    for i in concept_dict:
        # [relation, c1_vad, c1_c2_sim, weight, emotion_gap, score]   relation, weight, score
        rank_concept[i] = dict(sorted(concept_dict[i].items(), key=lambda x: x[1][5], reverse=True))  # 根据vad由大到小排序
        rank_concept[i] = [[l, concept_dict[i][l][0], concept_dict[i][l][1], concept_dict[i][l][2], concept_dict[i][l][3], concept_dict[i][l][4], concept_dict[i][l][5]] for l in concept_dict[i]]
    json.dump(rank_concept, rank_concept_file, indent=4)

def read_concept_vad(data_dir,dataset,data_train,data_val,data_test,vocab):
    def wordCate(word_pos):
        w_p = get_wordnet_pos(word_pos[1])
        if w_p == wordnet.NOUN or w_p == wordnet.ADV or w_p == wordnet.ADJ or w_p == wordnet.VERB:
            return True
        else:
            return False
    
    def augmentation(VAD, concept, data, word2index, concept_num):
        data = deepcopy(data)
        data["concepts"] = []
        data["sample_concepts"] = []
        data["vads"] = []
        data["vad"] = []
        data["target_vad"] = []
        data["target_vads"] = []
        contexts = data["context"]
        for i, sample in tqdm(enumerate(contexts)):
            vads = []  # each item is sentence, each sentence contains a list word' vad vectors
            vad = []
            concepts = []  # concepts of each sample
            total_concepts = []
            total_concepts_tid = []
            for j, sentence in enumerate(sample):
                words_pos = nltk.pos_tag(sentence)

                vads.append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in sentence])
                vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

                sentence_concepts = [
                    concept[word] if word in word2index and word not in stop_words and word in concept and wordCate(words_pos[wi]) else []
                    for wi, word in enumerate(sentence)]

                sentence_concept_words = []  # for each sentence
                sentence_concept_vads = []
                sentence_concept_vad = []

                for cti, uc in enumerate(sentence_concepts):  # filter concepts of each token, complete their VAD value, select top total_concept_num.
                    concept_words = []  # for each token
                    concept_vads = []
                    concept_vad = []
                    if uc != []:  # this token has concepts
                        for c in uc:  # iterate the concept lists [c,r,w] of each token
                            if c[1] not in REMOVE_RELATIONS and c[0] not in stop_words and c[0] in word2index:   # remove concpets that are stopwords or not in the dict
                                if c[0] in VAD and emotion_intensity(VAD, c[0]) >= 0.6:
                                    concept_words.append(c[0])
                                    concept_vads.append(VAD[c[0]])
                                    concept_vad.append(emotion_intensity(VAD, c[0]))
                                    total_concepts.append(c[0])  # all concepts of a sentence
                                    total_concepts_tid.append([j,cti])  # the token that each concept belongs to

                        # concept_words = concept_words[:5]
                        # concept_vads = concept_vads[:5]
                        # concept_vad = concept_vad[:5]
                        concept_words = concept_words[:concept_num]
                        concept_vads = concept_vads[:concept_num]
                        concept_vad = concept_vad[:concept_num]

                    sentence_concept_words.append(concept_words)
                    sentence_concept_vads.append(concept_vads)
                    sentence_concept_vad.append(concept_vad)

                sentence_concepts = [sentence_concept_words, sentence_concept_vads, sentence_concept_vad]
                concepts.append(sentence_concepts)
            data['concepts'].append(concepts)
            data['sample_concepts'].append([total_concepts, total_concepts_tid])
            data['vads'].append(vads)
            data['vad'].append(vad)
        
        targets = data['target']
        for i, target in enumerate(targets):
            # each item is the VAD info list of each target token
            data['target_vads'].append([VAD[word] if word in word2index and word in VAD else [0.5, 0.0, 0.5] for word in target])
            data['target_vad'].append([emotion_intensity(VAD, word) if word in VAD and word in word2index else 0.0 for word in target])
        print("Concept Process Finish......")
        assert len(data["context"])==len(data["context_role"])==len(data["situation"])==len(data["emotion"])==len(data["emotion_context"])==len(data["target"])==len(data["utt_cs"])==len(data["utt_cs_split"])==len(data["situation_cs"])==len(data["concepts"])==len(data["sample_concepts"])==len(data["vad"])==len(data["vads"])==len(data["target_vad"])==len(data["target_vads"])
        return data
        
    VAD = json.load(open(f"{data_dir}/VAD.json","r",encoding="utf-8"))
    concept = json.load(open(f"{data_dir}/{dataset}/ConceptNet_VAD_dict.json","r",encoding="utf-8"))
    data_train = augmentation(VAD, concept, data_train, vocab.word2index, config.concept_num)
    data_val = augmentation(VAD, concept, data_val, vocab.word2index, config.concept_num)
    data_test = augmentation(VAD, concept, data_test, vocab.word2index, config.concept_num)
    return data_train, data_val, data_test, vocab

def load_dataset():
    data_dir = config.data_dir
    dataset = config.dataset
    cache_file = f"{data_dir}/{dataset}/dataset_preproc.p"
    comet_cache_file = f"{data_dir}/{dataset}/dataset_comet_preproc.p"
    conceptnet_file = f"{data_dir}/{dataset}/ConceptNet.json"
    if os.path.exists(cache_file):
        print("Loading ", dataset, " Dataset......")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building Dataset......")
        vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                    config.KG_idx: "KG",
                    config.SEP_idx: "SEP"
                }
        )
        if dataset == "ESConv":
            if not os.path.exists(comet_cache_file):
                data_file = f"{data_dir}/{dataset}/ESConv.json"
                data_tra, data_val, data_tst, vocab = read_files(dataset=dataset, data_file=data_file, vocab=vocab)
                with open(comet_cache_file, "wb") as f:
                    pickle.dump([data_tra, data_val, data_tst, vocab], f)
                    print("SAVE COMET PiCKLE")
            else:
                with open(comet_cache_file, "rb") as f:
                    [data_tra, data_val, data_tst, vocab] = pickle.load(f)
            if not os.path.exists(conceptnet_file):
                get_concept_dict(data_dir, dataset, vocab)
                rank_concept_dict(data_dir, dataset)
            
            data_tra, data_val, data_tst, vocab = read_concept_vad(data_dir, dataset, data_train=data_tra, data_val=data_val, data_test=data_tst, vocab=vocab)
        elif dataset == "ED":
            if not os.path.exists(comet_cache_file):
                data_tra, data_val, data_tst, vocab = read_files(dataset=dataset, data_file=None, vocab=vocab)
                with open(comet_cache_file, "wb") as f:
                    pickle.dump([data_tra, data_val, data_tst, vocab], f)
                    print("SAVE COMET PiCKLE")
            else:
                with open(comet_cache_file, "rb") as f:
                    [data_tra, data_val, data_tst, vocab] = pickle.load(f)
            if not os.path.exists(conceptnet_file):
                get_concept_dict(data_dir, dataset, vocab)
                rank_concept_dict(data_dir, dataset)
            data_tra, data_val, data_tst, vocab = read_concept_vad(data_dir, dataset, data_train=data_tra, data_val=data_val, data_test=data_tst, vocab=vocab)
        else:
            raise ValueError(dataset + " does not exist!")
        
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Save PICKLE!")
    
    for i in range(3):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.esc_emo_map = esc_emo_map
        self.esc_strategy_map = esc_strategy_map
        self.ed_emo_map = ed_emo_map
        self.dataset = config.dataset
        self.analyzer = SentimentIntensityAnalyzer()
        
    def __len__(self):
        return len(self.data["target"])
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        # context and concept
        item["context_text"] = self.data["context"][index]
        item["context_role"] = self.data["context_role"][index]
        inputs = self.preprocess([self.data["context"][index],
                                  self.data["vads"][index],
                                  self.data["vad"][index],
                                  self.data["concepts"][index],
                                  self.data["context_role"][index]],
                                 concept=True)
        item["context"], item["context_ext"], item["context_mask"], item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], item["concept_vad"], \
        item["oovs"]= inputs
        
        item["situation_text"] = self.data["situation"][index]
        
        # emotion
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )
        if self.dataset == "ESConv":
            item["emotion"], item["emotion_label"] = self.preprocess_emo(
                item["emotion_text"], self.esc_emo_map
            )  
        elif self.dataset == "ED":
            item["emotion"], item["emotion_label"] = self.preprocess_emo(
                item["emotion_text"], self.ed_emo_map
            )
        (
            item["emotion_context"],
            item["emotion_context_mask"], # mask 有问题，但不用该mask
        ) = self.preprocess([item["emotion_context"], item["context_role"]])
        
        # strategy
        if self.dataset == "ESConv":
            item["context_strategy_seqs"] = self.data["context_strategy_seqs"][index]
            item["strategy_text"] = self.data["strategy"][index]
            item["strategy_seqs"] = self.preprocess(item["context_strategy_seqs"], strategy=True)
            item["strategy"], item["strategy_label"] = self.preprocess_strategy(
                item["strategy_text"], self.esc_strategy_map
            )
               
        # target
        item["target_text"] = self.data["target"][index]
        item["target"] = self.preprocess(item["target_text"], anw=True)
        
        # target posterior
        item["target_post"], item["target_post_mask"] = self.preprocess([[item["target_text"]], ["speaker"]])
        
        assert len(item["target"][:-1]) == len(item["target_post"][1:])
        
        # commonsense, last uttr of seeker
        # relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
        item["cs_text"] = self.data["utt_cs"][index]
        item["uttr_text"] = item["cs_text"]["uttr"]
        item["x_intent_txt"] = item["cs_text"]["xIntent"]
        item["x_need_txt"] = item["cs_text"]["xNeed"]
        item["x_want_txt"] = item["cs_text"]["xWant"]
        item["x_effect_txt"] = item["cs_text"]["xEffect"]
        item["x_react_txt"] = item["cs_text"]["xReact"]
        
        item["uttr"], item["uttr_mask"] = self.preprocess([[item["uttr_text"]], ["seeker"]])
        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")
        item["uttr_react"] = self.preprocess_react(item["x_react_txt"])
        
        # commonsense split, last uttr of seeker is split 
        item["cs_split_text"] = self.data["utt_cs_split"][index]
        item["uttr_split_text"] = [cs_split["uttr"] for cs_split in item["cs_split_text"]]
        item["x_intent_split_txt"] = [cs_split["xIntent"] for cs_split in item["cs_split_text"]]
        item["x_need_split_txt"] = [cs_split["xNeed"] for cs_split in item["cs_split_text"]]
        item["x_want_split_txt"] = [cs_split["xWant"] for cs_split in item["cs_split_text"]]
        item["x_effect_split_txt"] = [cs_split["xEffect"] for cs_split in item["cs_split_text"]]
        item["x_react_split_txt"] = [cs_split["xReact"] for cs_split in item["cs_split_text"]]
        
        uttr_split = [(self.preprocess([[uttr],["seeker"]])) for uttr in item["uttr_split_text"]]
        item["uttr_split"] = [uttr_ids for uttr_ids, uttr_masks in uttr_split]
        item["uttr_split_mask"] = [uttr_masks for uttr_ids, uttr_masks in uttr_split]
        item["x_intent_split"] = [self.preprocess(intent, cs=True) for intent in item["x_intent_split_txt"]]
        item["x_need_split"] = [self.preprocess(need, cs=True) for need in item["x_need_split_txt"]]
        item["x_want_split"] = [self.preprocess(want, cs=True) for want in item["x_want_split_txt"]]
        item["x_effect_split"] = [self.preprocess(effect, cs=True) for effect in item["x_effect_split_txt"]]
        item["x_react_split"] = [self.preprocess(react, cs="react") for react in item["x_react_split_txt"]]
        item["uttr_split_react"] = [torch.LongTensor(self.preprocess_react(react)) for react in item["x_react_split_txt"]]
        assert len(item["uttr_split"]) == len(item["uttr_split_mask"]) == len(item["x_intent_split"]) == len(item["x_need_split"]) == len(item["x_want_split"]) == len(item["x_effect_split"]) == len(item["x_react_split"])
        
        return item
       
    def preprocess_react(self, reacts):
        sequences = []
        for sent in reacts:
            if sent == ["none"]:
                continue
            for word in sent:
                if word in self.vocab.word2index and word not in ["none","to"]:
                    if self.vocab.word2index[word] not in sequences:
                        sequences.append(self.vocab.word2index[word])
        return sequences
     
    def process_oov(self, context, concept):  #
        ids = []
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:
                    if w not in oovs:
                        oovs.append(w)
                    oov_num = oovs.index(w)
                    ids.append(len(self.vocab.word2index) + oov_num)

        for sentence_concept in concept:
            for token_concept in sentence_concept:
                for c in token_concept:
                    if c not in oovs and c not in self.vocab.word2index:
                        oovs.append(c)
        return ids, oovs
    
    def preprocess(self, arr, anw=False, cs=None, strategy=False, concept=False):
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            # sequences = [config.CLS_idx] if cs != "react" else []
            sequences = []
            for sent in arr:
                if sent == ["none"]:
                    continue
                sequences.append([
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["none"]
                ])
                if cs != "react":
                    sequences[-1] = [config.CLS_idx]+sequences[-1]
            # while len(sequences)<config.cs_num:
            #     sequences.append([config.PAD_idx])
            # padding_seq = merge(sequences)
            return sequences
        elif strategy:
            sequence = [config.CLS_idx]
            sequence.extend([
                self.esc_strategy_map[word]
                for word in arr
                if word != "none"
            ])
            return torch.LongTensor(sequence)
        elif concept:
            context = arr[0]
            context_vads = arr[1]
            context_vad = arr[2]
            concept = [arr[3][l][0] for l in range(len(arr[3]))]
            concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
            concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]
            context_role = arr[4]

            X_dial = [config.CLS_idx]
            X_dial_ext = [config.CLS_idx]
            X_mask = [config.CLS_idx]  # for dialogue state
            X_vads = [[0.5, 0.0, 0.5]]
            X_vad = [0.0]

            X_concept_text = defaultdict(list)
            X_concept = [[]]  # 初始值是cls token
            X_concept_ext = [[]]
            X_concept_vads = [[0.5, 0.0, 0.5]]
            X_concept_vad = [0.0]
            assert len(context) == len(concept)

            X_ext, X_oovs = self.process_oov(context, concept)
            X_dial_ext += X_ext

            for (i, sentence), role in zip(enumerate(context), context_role):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if role == "seeker" else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
                X_vads += context_vads[i]
                X_vad += context_vad[i]

                for j, token_conlist in enumerate(concept[i]):
                    if token_conlist == []:
                        X_concept.append([])
                        X_concept_ext.append([])
                        X_concept_vads.append([0.5, 0.0, 0.5])  # ??
                        X_concept_vad.append(0.0)
                    else:
                        X_concept_text[sentence[j]] += token_conlist[:config.concept_num]
                        X_concept.append([self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]])

                        con_ext = []
                        for con_word in token_conlist[:config.concept_num]:
                            if con_word in self.vocab.word2index:
                                con_ext.append(self.vocab.word2index[con_word])
                            else:
                                if con_word in X_oovs:
                                    con_ext.append(X_oovs.index(con_word) + len(self.vocab.word2index))
                                else:
                                    con_ext.append(config.UNK_idx)
                        X_concept_ext.append(con_ext)
                        X_concept_vads.append(concept_vads[i][j][:config.concept_num])
                        X_concept_vad.append(concept_vad[i][j][:config.concept_num])

                        assert len([self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]]) == len(concept_vads[i][j][:config.concept_num]) == len(concept_vad[i][j][:config.concept_num])
            assert len(X_dial) == len(X_mask) == len(X_concept) == len(X_concept_vad) == len(X_concept_vads)

            return X_dial, X_dial_ext, X_mask, X_vads, X_vad, \
                   X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad, \
                   X_oovs
        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for sentence, role in zip(arr[0], arr[-1]):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if role == "seeker"
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)
             
    
    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]
    
    def preprocess_strategy(self, strategy, strategy_map):
        program = [0] * len(strategy_map)
        program[strategy_map[strategy]] = 1
        return program, strategy_map[strategy]
    
    def collate_fn(self, data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(
                len(sequences), max(lengths)
            ).long()  ## padding index 1
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
            return padded_seqs, lengths
        
        def merge_vad(vads_sequences, vad_sequences):  # for context
            lengths = [len(seq) for seq in vad_sequences]
            padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
            padding_vad = torch.FloatTensor([[0.0]]).repeat(len(vads_sequences), max(lengths)) # 原本为0.5

            for i, vads in enumerate(vads_sequences):
                end = lengths[i]  # the length of context
                padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
                padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
            return padding_vads, padding_vad  # (bsz, max_context_len, 3); (bsz, max_context_len)
        
        def merge_concept(samples, samples_ext, samples_vads, samples_vad):
            concept_lengths = []  # 每个sample的concepts数目
            token_concept_lengths = []  # 每个sample的每个token的concepts数目
            concepts_list = []
            concepts_ext_list = []
            concepts_vads_list = []
            concepts_vad_list = []

            for i, sample in enumerate(samples):
                length = 0  # 记录当前样本总共有多少个concept，
                sample_concepts = []
                sample_concepts_ext = []
                token_length = []
                vads = []
                vad = []

                for c, token in enumerate(sample):
                    if token == []:  # 这个token没有concept
                        token_length.append(0)
                        continue
                    length += len(token)
                    token_length.append(len(token))
                    sample_concepts += token
                    sample_concepts_ext += samples_ext[i][c]
                    vads += samples_vads[i][c]
                    vad += samples_vad[i][c]

                if length > config.total_concept_num:
                    value, rank = torch.topk(torch.LongTensor(vad), k=config.total_concept_num)

                    new_length = 1
                    new_sample_concepts = [config.SEP_idx]  # for each sample
                    new_sample_concepts_ext = [config.SEP_idx]
                    new_token_length = []
                    new_vads = [[0.5,0.0,0.5]]
                    new_vad = [0.0]

                    cur_idx = 0
                    for ti, token in enumerate(sample):
                        if token == []:
                            new_token_length.append(0)
                            continue
                        top_length = 0
                        for ci, con in enumerate(token):
                            point_idx = cur_idx + ci
                            if point_idx in rank:
                                top_length += 1
                                new_length += 1
                                new_sample_concepts.append(con)
                                new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                                new_vads.append(samples_vads[i][ti][ci])
                                new_vad.append(samples_vad[i][ti][ci])
                                assert len(samples_vads[i][ti][ci]) == 3

                        new_token_length.append(top_length)
                        cur_idx += len(token)

                    new_length += 1  # for sep token
                    new_sample_concepts = [config.SEP_idx] + new_sample_concepts
                    new_sample_concepts_ext = [config.SEP_idx] + new_sample_concepts_ext
                    new_vads = [[0.5,0.0,0.5]] + new_vads
                    new_vad = [0.0] + new_vad

                    concept_lengths.append(new_length)  # the number of concepts including SEP
                    token_concept_lengths.append(new_token_length)  # the number of tokens which have concepts
                    concepts_list.append(new_sample_concepts)
                    concepts_ext_list.append(new_sample_concepts_ext)
                    concepts_vads_list.append(new_vads)
                    concepts_vad_list.append(new_vad)
                    assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(new_sample_concepts_ext), "The number of concept tokens, vads [*,*,*], and vad * should be the same."
                    assert len(new_token_length) == len(token_length)
                else:
                    length += 1
                    sample_concepts = [config.SEP_idx] + sample_concepts
                    sample_concepts_ext = [config.SEP_idx] + sample_concepts_ext
                    vads = [[0.5,0.0,0.5]] + vads
                    vad = [0.0] + vad

                    concept_lengths.append(length)
                    token_concept_lengths.append(token_length)
                    concepts_list.append(sample_concepts)
                    concepts_ext_list.append(sample_concepts_ext)
                    concepts_vads_list.append(vads)
                    concepts_vad_list.append(vad)

            if max(concept_lengths) != 0:
                padded_concepts = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len); add 1 for root
                padded_concepts_ext = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len)
                padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths), 1) ## padding index 1 (bsz, max_concept_len)
                padded_concepts_vad = torch.FloatTensor([[0.0]]).repeat(len(samples), max(concept_lengths))  ## padding index 1 (bsz, max_concept_len)
                padded_mask = torch.ones(len(samples), max(concept_lengths)).long()  # concept(dialogue) state

                for j, concepts in enumerate(concepts_list):
                    end = concept_lengths[j]
                    if end == 0:
                        continue
                    padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                    padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                    padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                    padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                    padded_mask[j, :end] = config.KG_idx  # for DIALOGUE STATE

                return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
            else:  # there is no concept in this mini-batch
                return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])
        
        def concept_adj_mask(context, context_lengths, concepts, token_concept_lengths):
            '''
            :param self:
            :param context: (bsz, max_context_len)
            :param context_lengths: [] len=bsz
            :param concepts: (bsz, max_concept_len)
            :param token_concept_lengths: [] len=bsz;
            :return:
            '''
            bsz, max_context_len = context.size()
            max_concept_len = concepts.size(1)  # include sep token
            adjacency_size = max_context_len + max_concept_len
            adjacency = torch.ones(bsz, max_context_len, adjacency_size)   ## todo padding index 1, 1=True

            for i in range(bsz):
                # ROOT -> TOKEN
                adjacency[i, 0, :context_lengths[i]] = 0
                adjacency[i, :context_lengths[i], 0] = 0

                con_idx = max_context_len+1       # add 1 because of sep token
                for j in range(context_lengths[i]):
                    adjacency[i, j, j - 1] = 0 # TOEKN_j -> TOKEN_j-1

                    token_concepts_length = token_concept_lengths[i][j]
                    if token_concepts_length == 0:
                        continue
                    else:
                        adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                        adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                        con_idx += token_concepts_length
            return adjacency
        
        def con_adj_mask(context, context_lengths, concepts, token_concept_lengths):
            '''
            :param self:
            :param context: (bsz, max_context_len)
            :param context_lengths: [] len=bsz
            :param concepts: (bsz, max_concept_len)
            :param token_concept_lengths: [] len=bsz;
            :return:
            '''
            bsz, max_context_len = context.size()
            max_concept_len = concepts.size(1)  # include sep token
            adjacency_size = max_context_len + max_concept_len
            adjacency = torch.ones(bsz, adjacency_size, adjacency_size)   ## todo padding index 1, 1=True

            for i in range(bsz):
                
                # Self Loop
                for j in range(adjacency_size):
                    adjacency[i, j, j] = 0
            
                # ROOT -> TOKEN
                adjacency[i, 0, :context_lengths[i]] = 0
                adjacency[i, :context_lengths[i], 0] = 0

                con_idx = max_context_len+1       # add 1 because of sep token
                for j in range(context_lengths[i]):
                    adjacency[i, j, j - 1] = 0 # TOEKN_j -> TOKEN_j-1
                    adjacency[i, j - 1, j] = 0 # TOEKN_j-1 -> TOKEN_j

                    token_concepts_length = token_concept_lengths[i][j]
                    if token_concepts_length == 0:
                        continue
                    else:
                        adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0
                        adjacency[i, con_idx:con_idx+token_concepts_length, j] = 0
                        
                        adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0
                        adjacency[i, con_idx:con_idx+token_concepts_length, 0] = 0
                        
                        con_idx += token_concepts_length
            return adjacency
        
        def merge_cs(sequences):
            lengths = [len(sub_seq) for seqs in sequences for sub_seq in seqs]
            nums = [len(seqs) for seqs in sequences]
            padded_seqs = torch.ones(
                len(sequences), max(nums), max(lengths)
            ).long()
            mask_seqs = torch.zeros(
                len(sequences), max(nums)
            ).long()
            for i, seqs in enumerate(sequences):
                for j, sub_seq in enumerate(seqs):
                    end  = len(sub_seq)
                    padded_seqs[i, j, :end] = torch.LongTensor(sub_seq[:end])
                
                mask_end = len(seqs)
                mask_seqs[i, :mask_end] = 1
            return padded_seqs, nums, mask_seqs
        
        def merge_uttr_split(sequences):
            split_lengths = [len(seqs) for seqs in sequences]
            seq_lengths = [len(sub_seq) for seqs in sequences for sub_seq in seqs]
            padded_seqs = torch.ones(
                len(sequences), max(split_lengths), max(seq_lengths)
            ).long()
            mask_seqs = torch.zeros(
                len(sequences), max(split_lengths)
            ).long()
            for i, seqs in enumerate(sequences):
                for j, sub_seq in enumerate(seqs):
                    end = len(sub_seq)
                    padded_seqs[i,j,:end] = sub_seq[:end]
                
                mask_end = len(seqs)
                mask_seqs[i, :mask_end] = 1
            return padded_seqs, split_lengths, mask_seqs
        
        def merge_cs_split(sequences):
            split_lengths = [len(seqs) for seqs in sequences]
            seq_nums = [len(sub_seq) for seqs in sequences for sub_seq in seqs]
            seq_lengths = [len(seq) for seqs in sequences for sub_seq in seqs for seq in sub_seq]
            padded_seqs = torch.ones(
                len(sequences), max(split_lengths), max(seq_nums), max(seq_lengths)
            ).long()
            mask_seqs = torch.zeros(
                len(sequences), max(split_lengths), max(seq_nums)
            ).long()
            for i, seqs in enumerate(sequences):
                for j, sub_seq in enumerate(seqs):
                    for k, seq in enumerate(sub_seq):
                        end = len(seq)
                        padded_seqs[i,j,k,:end] = torch.LongTensor(seq[:end])
                    
                    mask_end = len(sub_seq)
                    mask_seqs[i, j, :mask_end] = 1
            return padded_seqs, seq_nums, mask_seqs
        
        def cs_adj_mask(uttr, uttr_cs, uttr_split, uttr_split_cs):
            bsz = uttr.size(0)
            x_intent_nums = uttr_cs[1]
            x_need_nums = uttr_cs[3]
            x_want_nums = uttr_cs[5]
            x_effect_nums = uttr_cs[7]
            split_nums = uttr_split
            x_intent_split_nums = uttr_split_cs[1]
            x_need_split_nums = uttr_split_cs[3]
            x_want_split_nums = uttr_split_cs[5]
            x_effect_split_nums = uttr_split_cs[7]
            
            adjacency_size = 1 + max(split_nums) + max(x_intent_nums) + max(x_need_nums) + max(x_want_nums) + max(x_effect_nums) + max(split_nums)*max(x_intent_split_nums) + max(split_nums)*max(x_need_split_nums) + max(split_nums)*max(x_want_split_nums) + max(split_nums)*max(x_effect_split_nums)
            cs_adjacency = torch.ones(bsz, adjacency_size, adjacency_size).long()
            
            x_split_offset = 0
            
            for i in range(bsz):
                # self loop relation
                for j in range(adjacency_size):
                    cs_adjacency[i, j, j] = config.self_loop
                # uttr -> split relation -> uttr_split
                start = 1
                cs_adjacency[i, 0, start:start+split_nums[i]] = config.contain
                cs_adjacency[i, start:start+split_nums[i], 0] = config.contain
                # uttr_split -> temporary relation -> other uttr_split
                for j in range(split_nums[i]):
                    if j < split_nums[i]-1:
                        cs_adjacency[i, 1+j, 1+j+1:1+split_nums[i]] = config.temporary
                        cs_adjacency[i, 1+j+1:1+split_nums[i], 1+j] = config.temporary
                # uttr -> intent relation -> x_intent
                start = 1+max(split_nums)
                cs_adjacency[i, 0, start: start+x_intent_nums[i]] = config.intent_idx
                cs_adjacency[i, start: start+x_intent_nums[i], 0] = config.intent_idx
                # uttr -> need relation -> x_need
                start = 1+max(split_nums)+max(x_intent_nums)
                cs_adjacency[i, 0, start:start+x_need_nums[i]] = config.need_idx
                cs_adjacency[i, start:start+x_need_nums[i], 0] = config.need_idx
                # uttr -> want relation -> x_want
                start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)
                cs_adjacency[i, 0, start:start+x_want_nums[i]] = config.want_idx
                cs_adjacency[i, start:start+x_want_nums[i], 0] = config.want_idx
                # uttr -> effect relation -> x_effect
                start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)+max(x_want_nums)
                cs_adjacency[i, 0, start:start+x_effect_nums[i]] = config.effect_idx
                cs_adjacency[i, start:start+x_effect_nums[i], 0] = config.effect_idx
                
                x_offset = 0
                for split_uttr_idx in range(1, split_nums[i]+1):
                    # uttr_split -> intent relation -> x_intent_split
                    intent_start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)+max(x_want_nums)+max(x_effect_nums)+x_offset*max(x_intent_split_nums)
                    intent_end = intent_start + x_intent_split_nums[x_split_offset]
                    cs_adjacency[i, split_uttr_idx, intent_start:intent_end] = config.intent_idx
                    cs_adjacency[i, intent_start:intent_end, split_uttr_idx] = config.intent_idx
                    
                    # uttr_split -> need relation -> x_need_split
                    need_start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)+max(x_want_nums)+max(x_effect_nums)+max(split_nums)*max(x_intent_split_nums)+x_offset*max(x_need_split_nums)
                    need_end = need_start + x_need_split_nums[x_split_offset]
                    cs_adjacency[i, split_uttr_idx, need_start:need_end] = config.need_idx
                    cs_adjacency[i, need_start:need_end, split_uttr_idx] = config.need_idx
                    
                    # uttr_split -> want relation -> x_want_split
                    want_start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)+max(x_want_nums)+max(x_effect_nums)+max(split_nums)*max(x_intent_split_nums)+max(split_nums)*max(x_need_split_nums)+x_offset*max(x_want_split_nums)
                    want_end = want_start + x_want_split_nums[x_split_offset]
                    cs_adjacency[i, split_uttr_idx, want_start:want_end] = config.want_idx
                    cs_adjacency[i, want_start:want_end, split_uttr_idx] = config.want_idx
                    
                    # uttr_split -> effect relation -> x_effect_split
                    effect_start = 1+max(split_nums)+max(x_intent_nums)+max(x_need_nums)+max(x_want_nums)+max(x_effect_nums)+max(split_nums)*max(x_intent_split_nums)+max(split_nums)*max(x_need_split_nums)+max(split_nums)*max(x_want_split_nums)+x_offset*max(x_effect_split_nums)
                    effect_end = effect_start +  x_effect_split_nums[x_split_offset]
                    cs_adjacency[i, split_uttr_idx, effect_start:effect_end] = config.effect_idx
                    cs_adjacency[i, effect_start:effect_end, split_uttr_idx] = config.effect_idx
                    
                    x_offset += 1
                    x_split_offset += 1
            
            return cs_adjacency
        
        def merge_cs_batch(commonsense_list):
            info = [cs.size() for cs in commonsense_list]
            lengths = [length for _, _, length in info]
            new_commonsense_list = list()
            for cs in commonsense_list:
                if cs.size(2) < max(lengths):
                    padded_seqs = torch.ones(
                        cs.size(0), cs.size(1), max(lengths)-cs.size(2)
                    ).long()
                    cs = torch.cat((cs, padded_seqs), dim=-1)
                new_commonsense_list.append(cs)
            return torch.cat(new_commonsense_list, dim=1)
        
        data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]
        
        # input context
        input_batch, input_lengths = merge(item_info["context"])
        mask_input, mask_input_lengths = merge(item_info["context_mask"])
        
        # context vad
        context_vads_batch, context_vad_batch = merge_vad(item_info["vads"], item_info["vad"])
        assert input_batch.size(1) == context_vad_batch.size(1)
        
        # concept, vad, vads
        concept_inputs = merge_concept(item_info['concept'],
                                       item_info['concept_ext'],
                                       item_info["concept_vads"],
                                       item_info["concept_vad"])  # (bsz, max_concept_len)
        concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = concept_inputs
        
        # concept adjacency mask (bsz, max_context_len, max_context_len+max_concept_len)
        if concept_batch.size()[0] != 0:
            adjacency_mask_batch = concept_adj_mask(input_batch, input_lengths, concept_batch, token_concept_lengths)
            concept_adjacency_mask_batch = con_adj_mask(input_batch, input_lengths, concept_batch, token_concept_lengths)
        else:
            adjacency_mask_batch = torch.Tensor([])
            concept_adjacency_mask_batch = torch.Tensor([])
        
        # cs
        uttr_batch, uttr_lengths = merge(item_info["uttr"])
        uttr_mask_batch, _ = merge(item_info["uttr_mask"])
        x_intent_batch, x_intent_nums, x_intent_mask = merge_cs(item_info["x_intent"])
        x_need_batch, x_need_nums, x_need_mask = merge_cs(item_info["x_need"])
        x_want_batch, x_want_nums, x_want_mask = merge_cs(item_info["x_want"])
        x_effect_batch, x_effect_nums, x_effect_mask = merge_cs(item_info["x_effect"])
        uttr_react_batch, _ = merge(item_info["uttr_react"])
        
        uttr_split_batch, uttr_split_lengths, _ = merge_uttr_split(item_info["uttr_split"])
        uttr_split_mask_batch, _, _ = merge_uttr_split(item_info["uttr_split_mask"])
        x_intent_split_batch, x_intent_split_nums, x_intent_split_mask = merge_cs_split(item_info["x_intent_split"])
        x_need_split_batch, x_need_split_nums, x_need_split_mask = merge_cs_split(item_info["x_need_split"])
        x_want_split_batch, x_want_split_nums, x_want_split_mask = merge_cs_split(item_info["x_want_split"])
        x_effect_split_batch, x_effect_split_nums, x_effect_split_mask = merge_cs_split(item_info["x_effect_split"])
        uttr_split_react_batch, _, uttr_split_react_mask = merge_uttr_split(item_info["uttr_split_react"])
        
        cs_adjacency_mask_batch = cs_adj_mask(uttr_batch,
                                              (x_intent_batch, x_intent_nums,
                                               x_need_batch, x_need_nums,
                                               x_want_batch, x_want_nums,
                                               x_effect_batch, x_effect_nums),
                                              uttr_split_lengths,
                                              (x_intent_split_batch, x_intent_split_nums,
                                               x_need_split_batch, x_need_split_nums,
                                               x_want_split_batch, x_want_split_nums,
                                               x_effect_split_batch, x_effect_split_nums))
        
        uttr_batch_concat = merge_cs_batch([
            uttr_batch.unsqueeze(1),
            uttr_split_batch
        ])
        
        bsz, _, split_intent_num, intent_lengths = x_intent_split_batch.size()
        bsz, _, split_need_num, need_lengths = x_need_split_batch.size()
        bsz, _, split_want_num, want_lengths = x_want_split_batch.size()
        bsz, _, split_effect_num, effect_lengths = x_effect_split_batch.size()
        cs_batch = merge_cs_batch([
            x_intent_batch,
            x_need_batch,
            x_want_batch,
            x_effect_batch,
            x_intent_split_batch.view(bsz,-1,intent_lengths),
            x_need_split_batch.view(bsz,-1,need_lengths),
            x_want_split_batch.view(bsz,-1,want_lengths),
            x_effect_split_batch.view(bsz,-1,effect_lengths)])
        cs_mask = torch.cat((
            x_intent_mask,
            x_need_mask,
            x_want_mask,
            x_effect_mask,
            x_intent_split_mask.view(bsz, -1),
            x_need_split_mask.view(bsz, -1),
            x_want_split_mask.view(bsz, -1),
            x_effect_split_mask.view(bsz, -1)), dim=-1)
        
        react_batch = merge_cs_batch([
            uttr_react_batch.unsqueeze(1),
            uttr_split_react_batch
        ])
        max_uttr_cs_num = max(x_intent_nums) + max(x_need_nums) + max(x_want_nums) + max(x_effect_nums)
        max_uttr_split_num = max(x_intent_split_nums) + max(x_need_split_nums) + max(x_want_split_nums) + max(x_effect_split_nums)
        
        # emotion
        emotion_batch, emotion_lengths = merge(item_info["emotion_context"])
        emotion_batch_mask, emotion_mask_lengths = merge(item_info["emotion_context_mask"])
        # emotion_program, emotion_label = item_info["emotion"], item_info["emotion_label"]
        
        # strategy
        if self.dataset == "ESConv":
            strategy_seqs_batch, strategy_seqs_lengths = merge(item_info["strategy_seqs"])
            # strategy_program, strategy_label = item_info["strategy"], item_info["strategy_label"]
        
        # Target
        target_batch, target_lengths = merge(item_info["target"])
        
        target_post_batch, _ = merge(item_info["target_post"])
        target_post_mask, _ = merge(item_info["target_post_mask"])
        
        d = {}
        ## input, context
        d["input_batch"] = input_batch.to(config.device) # bsz, max_context_length
        d["input_lengths"] = torch.LongTensor(input_lengths).to(config.device) # bsz,
        d["mask_input"] = mask_input.to(config.device) # bsz, max_context_length
        d["context_vads"] = context_vads_batch.to(config.device) # bsz, max_context_length, 3
        d["context_vad"] = context_vad_batch.to(config.device) # bsz, max_context_length
        d["emotion_context_batch"] = emotion_batch.to(config.device)
        d["emotion_context_batch_mask"] = emotion_batch_mask.to(config.device)
        
        ## concept
        d["concept_batch"] = concept_batch.to(config.device) # bsz, max_concept_len
        d["concept_ext_batch"] = concept_ext_batch.to(config.device) # bsz, max_concept_len
        d["concept_lengths"] = torch.LongTensor(concept_lengths).to(config.device) # bsz,
        d["mask_concept"] = mask_concept.to(config.device) # bsz, max_concept_len
        d["concept_vads_batch"] = concepts_vads_batch.to(config.device) # bsz, max_concept_len, 3
        d["concept_vad_batch"] = concepts_vad_batch.to(config.device) # bsz, max_concept_len
        d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(config.device)
        d["concept_adjacency_mask_batch"] = concept_adjacency_mask_batch.bool().to(config.device)
        
        ## cs
        d["uttr_batch"] = uttr_batch.to(config.device) # bsz, max_lengths
        d["uttr_mask_batch"] = uttr_mask_batch.to(config.device) # bsz, max_lengths
        d["x_intent_batch"] = x_intent_batch.to(config.device) # bsz, max_nums, max_lengths
        d["x_intent_mask"] = x_intent_mask.to(config.device) # bsz, max_nums
        d["x_need_batch"] = x_need_batch.to(config.device) # bsz, max_nums, max_lengths
        d["x_need_mask"] = x_need_mask.to(config.device) # bsz, max_nums
        d["x_want_batch"] = x_want_batch.to(config.device) # bsz, max_nums, max_lengths
        d["x_want_mask"] = x_want_mask.to(config.device) # bsz, max_nums
        d["x_effect_batch"] = x_effect_batch.to(config.device) # bsz, max_nums, max_lengths
        d["x_effect_mask"] = x_effect_mask.to(config.device) # bsz, max_nums
        
        d["uttr_split_batch"] = uttr_split_batch.to(config.device) # bsz, max_split_num, max_lengths
        d["uttr_split_mask_batch"] = uttr_split_mask_batch.to(config.device) # bsz, max_split_num, max_lengths
        d["x_intent_split_batch"] = x_intent_split_batch.to(config.device) # bsz, max_split_num, max_num, max_lengths
        d["x_intent_split_mask"] = x_intent_split_mask.to(config.device) # bsz, max_split_num, max_num
        d["x_need_split_batch"] = x_need_split_batch.to(config.device) # bsz, max_split_num, max_num, max_lengths
        d["x_need_split_mask"] = x_need_split_mask.to(config.device) # bsz, max_split_num, max_num
        d["x_want_split_batch"] = x_want_split_batch.to(config.device) # bsz, max_split_num, max_num, max_lengths
        d["x_want_split_mask"] = x_want_split_mask.to(config.device) # bsz, max_split_num, max_num
        d["x_effect_split_batch"] = x_effect_split_batch.to(config.device) # bsz, max_split_num, max_num, max_lengths
        d["x_effect_split_mask"] = x_effect_split_mask.to(config.device) # bsz, max_split_num, max_num
        
        d["cs_adjacency_mask_batch"] = cs_adjacency_mask_batch.to(config.device) # bsz, max_content_num, max_content_num
        
        d["uttr_batch_concat"] = uttr_batch_concat.to(config.device)
        d["cs_batch"] = cs_batch.to(config.device)
        d["cs_mask"] = cs_mask.to(config.device)
        d["react_batch"] = react_batch.to(config.device)
        d["max_uttr_cs_num"] = max_uttr_cs_num
        d["max_uttr_cs_split_num"] = max_uttr_split_num
        d["split_intent_num"] = split_intent_num
        d["split_need_num"] = split_need_num
        d["split_want_num"] = split_want_num
        d["split_effect_num"] = split_effect_num
        d["uttr_split_react_mask"] = uttr_split_react_mask.to(config.device)
        
        ## strategy
        if self.dataset == "ESConv":
            d["strategy_seqs_batch"] = strategy_seqs_batch.to(config.device)
            d["strategy_program"] = torch.LongTensor(item_info["strategy"]).to(config.device)
            d["strategy_label"] = torch.LongTensor(item_info["strategy_label"]).to(config.device)
        
        ## Target
        d["target_batch"] = target_batch.to(config.device)
        d["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)
        
        d["target_post_batch"] = target_post_batch.to(config.device)
        d["target_post_mask"] = target_post_mask.to(config.device)
        
        ## program
        d["target_program"] = torch.LongTensor(item_info["emotion"]).to(config.device)
        d["program_label"] = torch.LongTensor(item_info["emotion_label"]).to(config.device)
        
        ##text
        d["input_txt"] = item_info["context_text"]
        d["target_txt"] = item_info["target_text"]
        d["program_txt"] = item_info["emotion_text"]
        d["situation_txt"] = item_info["situation_text"]

        d["context_emotion_scores"] = item_info["context_emotion_scores"]
        
        return d
        

def prepare_data_seq(batch_size):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))
    
    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
    )
    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset_valid.collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=dataset_test.collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.esc_emo_map) if config.dataset=="ESConv" else len(dataset_train.ed_emo_map),
        len(dataset_train.esc_strategy_map)
    )