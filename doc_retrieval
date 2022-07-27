# import libraries
import random
import torch  # machine learning
import os.path  # detects if file exists
import glob  # collects all files in path
from transformers import BertTokenizer, BertModel  # bert model
from tabulate import tabulate  # print 2d array
import time  # calculate number of seconds
import numpy as np  # 2d array
from tqdm import tqdm  # track progress
import argparse  # substitution
from sklearn.cluster import KMeans  # kmeans clustering
import gc # garbage collect

parser = argparse.ArgumentParser()
parser.add_argument('--root', default="/home/marie/Downloads/")
parser.add_argument('--embedding', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--mode', default="kmeans,6")
args = parser.parse_args()


# separate text into sentences and clean
def clean_text(filename):
    import spacy  # identify sentences
    import re  # replacecos_q_list
    nlp = spacy.load("en_core_web_sm")
    with open(filename, 'r') as f:
        text = f.read()

    doc, sentence_list = nlp(text), []

    for sent in doc.sents:
        if sent[0].is_title and sent[-1].is_punct:
            has_noun = 2
            has_verb = 1

            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    has_noun -= 1

                elif token.pos_ == "VERB":
                    has_verb -= 1

            if has_noun < 1 and has_verb < 1:
                sentence_list.append(sent.text)

    for i in range(0, len(sentence_list)):
        sentence_list[i] = sentence_list[i].strip()
        sentence_list[i] = re.sub("@\S+", "", sentence_list[i])
        sentence_list[i] = re.sub("#", "", sentence_list[i])
        sentence_list[i] = re.sub("\n", "", sentence_list[i])
        sentence_list[i] = re.sub("-", "", sentence_list[i])
        sentence_list[i] = re.sub("[\(\[].*?[\)\]]", "", sentence_list[i])
        if sentence_list[i].find(' ') == -1:
            sentence_list[i] = None

    sentence_list = [i for i in sentence_list if i]
    return sentence_list


# create class for dataset
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.sentences = clean_text(filename)
        new_filename = filename.split("/")
        new_filename = "".join([args.root, "sample_txt_pgs_clean", new_filename[-1]])

        with open(new_filename[-1], 'w') as f:
            for sentence in self.sentences:
                f.writelines(sentence + "\n")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = self.sentences[idx]
        return x


# create bert model
class BERT_Model(torch.nn.Module):
    def __init__(self, pre_trained, tokenizer):
        super(BERT_Model, self).__init__()
        self.bert = pre_trained
        self.tokenizer = tokenizer
        self.output_dim = self.bert.config.hidden_size

    def forward(self, x):
        model_output = self.bert(**x).last_hidden_state.detach()
        model_output = torch.mean(model_output, dim=1)
        return model_output

    def encode(self, sentences):
        tokenizes_sentences = self.tokenizer(sentences)
        tokenizes_sentences = tokenizes_sentences.to(device)
        return self.forward(tokenizes_sentences)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def BERT_tokenizer(batch):
    return bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)


# bert model
my_encoder = BERT_Model(bert_model, BERT_tokenizer)

# create embedding for files and store in pt file
if args.embedding:
    txt_list, file_embedding = [], []  # glob.glob(args.root + "sample_txt_pgs/*.txt")

    for file in txt_list:
        name = file.split('/')
        pt_file_name = args.root + "test_pt/" + name[-1].replace(".txt", ".pt")

        if os.path.exists(pt_file_name):
            continue

        dataset = my_dataset(file)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1,
                                                  drop_last=False, collate_fn=BERT_tokenizer)

        for x in data_loader:
            file_embedding.append(my_encoder(x))

        torch.save(file_embedding, pt_file_name)  # save in pt file

# args.debug = True
if args.debug:
    pt_list = ["/home/marie/Downloads/test_pt/Ahuja_pg1.pt",
               "/home/marie/Downloads/test_pt/Titone2002_pg1.pt",
               "/home/marie/Downloads/test_pt/Gill_pg1.pt",
               "/home/marie/Downloads/test_pt/Ramsey2003_pg1.pt",
               "/home/marie/Downloads/test_pt/Sommer2006_pg1.pt",
               "/home/marie/Downloads/test_pt/Isler2017_pg1.pt",
               "/home/marie/Downloads/test_pt/Johnson2012_pg1.pt",
               "/home/marie/Downloads/test_pt/Morris2011_pg1.pt",
               "/home/marie/Downloads/test_pt/Rice2003_pg1.pt",
               "/home/marie/Downloads/test_pt/Brook2015_pg16.pt"]
else:
    pt_list = glob.glob(args.root + "test_pt/*.pt")

t0 = time.time()

normalized, tup = [], []

# normalize sentences
for pt_file in tqdm(pt_list):
    sentence_num, pt_load = 0, torch.load(pt_file)

    for sent in pt_load:
        sentence_num = sentence_num + 1
        normalized.append(torch.nn.functional.normalize(sent, dim=0))
        tup.append(f"{sentence_num}" + pt_file)

r_list = []
mode, num_r = args.mode.split(",")

# average
# r = torch.nn.functional.normalize(sum(normalized), dim=0)
# r_list.append(r)

if r_list:
    pass
else:
    num_r = int(num_r)

    if mode == "kmeans":
        normalized_ = [norm.numpy() for norm in random.choices(normalized, k=min(100000, len(normalized)))]
        kmeans = KMeans(n_clusters=num_r, random_state=0, n_init=1, verbose=10).fit(normalized_)
        r_list = torch.nn.functional.normalize(torch.tensor(kmeans.cluster_centers_).float())

    # random
    if mode == "rand":
        for i in range(0, num_r):
            r_list.append(torch.nn.functional.normalize(torch.rand(768), dim=0))

gc.collect()
cos = torch.nn.CosineSimilarity(dim=0)

# similarity between sentences and r
cos_list = [torch.Tensor([cos(sent_, r_vec) for r_vec in r_list]) for sent_ in tqdm(normalized)]

t1 = time.time()
print("\nLoad Finished(", t1 - t0, "seconds )\n")  # display time

# input query
query_list = ["Cognitive science is the study of the human mind and brain", "Cognitive scientists study intelligence and behavior with a focus on how nervous systems represent, process, and transform information", "Cognitive science is an interdisciplinary field with contributors from various fields, including psychology, neuroscience, linguistics, philosophy of mind, computer science, anthropology and biology", "Artificial intelligence involves the study of cognitive phenomena in machines to implement aspects of human intelligence in computers", "Consciousness is the awareness of external objects and experiences within oneself", "One of the fundamental concepts of cognitive science is that thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures", "The cognitive sciences began as an intellectual movement in the 1950s, called the cognitive revolution", "Cognitive science has a diversity of outlooks and methods that researchers in different fields bring to the study of mind and intelligence", "The goal of cognitive science is to understand the representations and processes in our minds that underwrite these capacities", "The brain controls all functions of the body, interprets information from the outside world, and embodies the essence of the mind and soul"]

k = -1

for text in query_list:
    query, query_embedding = [], []
    similaritylist, result_list, num = [0, 0, 0, 0, 0], [], 5
    similarity_list, max_list, skip = [], [], 0

    query.append(text)

    # query.append(input("\nSearch: "))

    t0 = time.time()  # start time

    # create embedding for query
    data_loader = torch.utils.data.DataLoader(query, batch_size=1, shuffle=False, pin_memory=True, num_workers=1,
                                              drop_last=False, collate_fn=BERT_tokenizer)

    for x in data_loader:
        query_embedding.append(my_encoder(x))

    # cosine similarity between query and sentences: want 1

    if mode == "base":
        for pt_file in tqdm(pt_list):
            pt_load, sentence_num, similarity_ = torch.load(pt_file), 0, []

            try:
                for i in range(0, len(pt_load)):
                    similarity_.append(cos(pt_load[i], query_embedding[0].flatten()))  # calculate similarity for each sentence in file

                for similarity in similarity_:
                    sentence_num = sentence_num + 1

                    name = pt_file.split('/')
                    similarity_list.append("".join([f"{similarity}:::", name[-1], f"::sentence{sentence_num}"]))

                similarity_list.sort(reverse=True)
                similarity_list = similarity_list[0:num]

            except:
                continue


    else:
        q = torch.nn.functional.normalize(query_embedding[0], dim=1)
        q = q.flatten()

        # similarity between query and r
        cos_q_list = torch.Tensor([cos(q, r) for r in r_list])

        for k in tqdm(range(0, len(cos_list))):

            max_sim = -(cos_list[k] - cos_q_list).abs().max() + 1

            if max_sim <= similaritylist[4]:
                skip = skip + 1
                continue

            else:
                name = tup[k].split('/')
                similarity = cos(normalized[k], q)

                similaritylist.append(similarity)

                similarity_list.append("".join([f"{similarity}:::", name[-1], f"::sentence{name[0]}"]))

                similarity_list.sort(reverse=True)
                similarity_list = similarity_list[0:num]
                similaritylist.sort(reverse=True)
                similaritylist = similaritylist[0:num]

        print("\nskips:", skip)

    # return top 5 results: result number, file, page, sentence
    for i in range(0, num):
        file_to_return = similarity_list[i].split(":::")
        result_list.append(file_to_return[-1].replace(".pt", ".txt"))

    t1 = time.time()
    print("\nResults (", t1 - t0, "seconds ):\n")  # display time

    display = np.empty((0, 5), str)  # 2d array using numpy
    order = 0

    for txt in result_list:
        similarity = similarity_list[order]
        similarity = similarity[0:similarity.find(':')]  # similarity

        order = order + 1

        result = txt.split("::")

        result[1] = (result[1].replace(".txt", "")).replace("pg", "")  # result[1] = page number

        num = int(result[-1].replace("sentence", ""))  # result[-1] = sentence number

        with open(args.root + f"sample_txt_pgs_clean/{result[0]}", 'r') as f:
            sentence_list = f.readlines()
            sentence = sentence_list[num - 1]  # sentence

        name_ = result[0].split("_")
        page_ = name_[-1].replace(".txt", "")
        page_ = page_.replace("pg", "")  # page number
        name_ = name_[0] + ".txt"  # file name

        display = np.append(display, np.array([[f"{order}", similarity, name_, page_, sentence]]),
                            axis=0)  # append results

    print("Query:", query[0], "\n")
    print(tabulate(display, headers=["Result", "Similarity", "File", "Page", "Text"]))  # print with headers
    print("\n")
