# import libraries
import os
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
import gc  # garbage collect
from pdf_to_text import extract_pdfdir_text

parser = argparse.ArgumentParser()
# root is source data
parser.add_argument('--source', default=f"/home/{os.getlogin()}/daviesearch/")
parser.add_argument(
    '--data', default=f"/home/{os.getlogin()}/daviesearch_data/")
parser.add_argument('--noscan', action='store_true', default=False)
parser.add_argument('--query', default="")
parser.add_argument('--debug', action='store_true', default=True)
# optimized vs nonoptimized (base if no optimization,
parser.add_argument('--mode', default="base,0")
# kmeans, #clusters if want optimization)
parser.add_argument('--nest', action='store_true')
args = parser.parse_args()

# verify data directory exists
if not os.path.exists(args.data):
    os.mkdir(args.data)
if not os.path.exists(os.path.join(args.data, "txt")):
    os.mkdir(os.path.join(args.data, "txt"))
if not os.path.exists(os.path.join(args.data, "pt")):
    os.mkdir(os.path.join(args.data, "pt"))
if not os.path.exists(os.path.join(args.data, "txt_clean")):
    os.mkdir(os.path.join(args.data, "txt_clean"))


# SETTINGS
pt_limit = 30000		# max number of document embeddings to load into memory
filename_max_len = 40  # character length of filename column in results
pagenum_max_digits = 4  # character length of pagenum column in results
sentence_max_len = 75  # character length of sentence text column in results

# separate text into sentences and clean


def read_file(filename):

    sentence_list = []

    try:
        with open(filename, 'r') as f:
            sentence_list = (f.readlines())
    except:
        print("error reading sentence file", filename)

    return sentence_list


# create class for dataset
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.sentences = read_file(filename)
        new_filename = filename.split("/")
        new = os.path.join(args.data, "txt_clean", new_filename[-1])

        if not os.path.exists(os.path.join(args.data, "txt_clean")):
            os.mkdir(os.path.join(args.data, "txt_clean"))

        with open(new, 'w') as f:
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

print("\n"*50)

if not args.noscan:
    # extract text to data directory
    extract_pdfdir_text(args.source, os.path.join(args.data, "txt"))

	# create embedding for files and store in pt file
    pt_file_dir = os.path.join(args.data, "pt")

    if not os.path.exists(pt_file_dir):
        os.makedirs(pt_file_dir)

    txt_list, file_embedding = glob.glob(
        os.path.join(args.data, "txt", "*.txt")), []  # getting text files

    print("\n-- GENERATING EMBEDDINGS to", pt_file_dir)
    for file in tqdm(txt_list):
        file_embedding = []
        name = file.split('/')
        pt_file_name = os.path.join(
            args.data, "pt", name[-1].replace(".txt", ".pt"))

        if os.path.exists(pt_file_name):
            continue

        dataset = my_dataset(file)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1,
                                                  drop_last=False, collate_fn=BERT_tokenizer)

        for x in data_loader:
            file_embedding.append(my_encoder(x))

        torch.save(file_embedding, pt_file_name)  # save in pt file


pt_list = glob.glob(os.path.join(args.data, "pt", "*.pt"))[:pt_limit]

t0 = time.time()

normalized, tup = [], []

# normalize sentences
print("\n-- LOADING AND NORMALIZING SENTENCES")
for pt_file in tqdm(pt_list):
    sentence_num, pt_load = 0, torch.load(pt_file)
    for sent in pt_load:
        sentence_num = sentence_num + 1
        normalized.append(torch.nn.functional.normalize(sent, dim=0))
        tup.append(f"{sentence_num}" + pt_file)
        #print(f"{sentence_num}" + pt_file)

r_list = []
mode, num_r = args.mode.split(",")

# average
# r = torch.nn.functional.normalize(sum(normalized), dim=0)
# r_list.append(r)

if r_list:
    pass
else:
    num_r = int(num_r)

    # kmeans
    if mode == "kmeans":
        # normalized_ = [norm.numpy() for norm in random.choices(normalized, k=min(100000, len(normalized)))]
        normalized_ = [norm.half().numpy() for norm in normalized]
        kmeans = KMeans(n_clusters=num_r, random_state=0,
                        n_init=1, verbose=10).fit(normalized_)
        r_list = torch.nn.functional.normalize(
            torch.tensor(kmeans.cluster_centers_).float())

        if args.nest:
            kmeans.labels_ = np.array([v*num_r for v in kmeans.labels_])
            r_list = []
            for i in range(num_r):
                offset = i*num_r
                select = np.where(kmeans.labels_ == offset)[0]
                normalized_2 = [normalized_[i] for i in select]
                kmeans2 = KMeans(n_clusters=num_r, random_state=0,
                                 n_init=1, verbose=10).fit(normalized_2)
                kmeans2.labels_ = np.array(
                    [v + i*num_r for v in kmeans2.labels_])
                np.put(kmeans.labels_, select, kmeans2.labels_)
                r_list.append(torch.nn.functional.normalize(
                    torch.tensor(kmeans2.cluster_centers_).float()))

            r_list = torch.stack(r_list).view(
                num_r*num_r, normalized_[0].shape[-1])


gc.collect()
cos = torch.nn.CosineSimilarity(dim=1)

# similarity between sentences and r
t1 = time.time()
print("\nLoad Finished(", t1 - t0, "seconds )\n")  # display time

k = -1

query, query_embedding = [], []
similaritylist, result_list, num = [0, 0, 0, 0, 0], [], 5
similarity_list, max_list, skip = [], [], 0
wrong_skip = 0

query.append(args.query if len(args.query) > 0 else input("\nSearch: "))

t0 = time.time()  # start time

# create embedding for query
data_loader = torch.utils.data.DataLoader(query, batch_size=1, shuffle=False, pin_memory=True, num_workers=1,
											drop_last=False, collate_fn=BERT_tokenizer)

for x in data_loader:
	query_embedding.append(my_encoder(x))

# cosine similarity between query and sentences: want 1

if mode == "base":
	q = torch.nn.functional.normalize(query_embedding[0], dim=1)
	q = q.flatten()
	print("\n-- SEARCHING")
	for k in tqdm(range(0, len(normalized))):
		name = tup[k].split('/')

		similarity = cos(normalized[k], q)

		similarity_list.append(
			"".join([f"{similarity.numpy()[0]}:::", name[-1], f"::sentence{name[0]}"]))

		similarity_list.sort(reverse=True)
		similarity_list = similarity_list[0:num]

else:
	q = torch.nn.functional.normalize(query_embedding[0], dim=1)
	q = q.flatten()

	# similarity between query and r
	cos_q_list = torch.Tensor([cos(q, r) for r in r_list])
	k_idx = cos_q_list.argmax()

	print("\n-- SEARCHING")
	for k in tqdm(range(0, len(normalized))):
		if kmeans.labels_[k] != k_idx:
			skip = skip + 1
			continue
		name = tup[k].split('/')
		similarity = cos(normalized[k], q)

		similarity_list.append(
			"".join([f"{similarity.numpy()[0]}:::", name[-1], f"::sentence{name[0]}"]))

		similarity_list.sort(reverse=True)
		similarity_list = similarity_list[0:num]

	print("\nskips:", skip, "wrong_skip", wrong_skip)

# return top 5 results: result number, file, page, sentence
for i in range(0, num):
	file_to_return = similarity_list[i].split(":::")
	result_list.append(file_to_return[-1].replace(".pt", ".txt"))

t1 = time.time()
print("\n-- RESULTS (", t1 - t0, "seconds )")  # display time

display = np.empty((0, 5), str)  # 2d array using numpy
order = 0

for txt in result_list:
	similarity = similarity_list[order]
	similarity = similarity[0:similarity.find(':')]  # similarity

	order = order + 1

	result = txt.split("::")

	result[1] = (result[1].replace(".txt", "")).replace(
		"pg", "")  # result[1] = page number

	# result[-1] = sentence number
	num = int(result[-1].replace("sentence", ""))

	with open(os.path.join(args.data, "txt", result[0]), 'r') as f:
		sentence_list = f.readlines()

		sentence = sentence_list[num - 1]  # sentence

	name_ = result[0].split("_")
	page_ = name_[-1].replace(".txt", "")
	page_ = page_.replace("pg", "")  # page number
	name_ = name_[0] + ".txt"  # file name

	display = np.append(display, np.array([[f"{order}", similarity, name_, page_, sentence]]),
						axis=0)  # append results

for row in display:
	if len(row[2]) > filename_max_len:
		row[2] = row[2][:filename_max_len-3] + "..."
	if len(row[3]) > pagenum_max_digits:
		pass  # lets hope we never run into that...
	if len(row[4]) > sentence_max_len:
		row[4] = row[4][:sentence_max_len-3] + "..."

print("Query:", query[0], "\n")
# print with headers
print(tabulate(display, headers=[
		"Result", "Similarity", "File", "Page", "Text"]))
print("\n")
