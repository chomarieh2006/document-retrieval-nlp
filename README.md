# Data Retrieval via NLP

## About
A search engine to look for certain files from the user.
Using machine learning and PyTorch to develop a model that can utilize NLP to search for and return relevant texts and images to the user's query.
Employs semantic embedding (for meaning rather than word frequency) and convolutional neural networks to retrieve data.

## Result Highlights

### [Research Paper](Accelerating_Document_Retrieval_with_k_Means_Clustering.pdf)

The following are the results for the document retrieval acceleration process.

| Chart | What does this mean? |
| ------------- | ------------- |
| <img src="base_times.png" width=100% height=100%> | Searches with no optimization took 17 seconds to load/prepare and about 7 seconds to calculate the top 5 nearest neighbor vectors. While 7 seconds is considerably fast and the base mode is the 100% accurate, times will get  slower as the system is exposed to more data. |
| <img src="kmeans_times.png" width=100% height=100%>  | With acceleration through k-means clustering, query times have decreased as the number of centroids increased. Ideally, at some point the time will be essentially 0. However,there is a limit (in this case about 2 seconds) to the acceleration, due to a constant factor that cannot be reduced. Despite this, the overall wait time is reduced significantly.|
| <img src="kmeans_load.png" width=100% height=100%> | However, optimization through k-means clustering comes as a trade-off with loading times. As the number of centroids increases, the loading times increase with it. In this case, as k increases from 1 to 32, the loading time increases from 25 seconds to 125 seconds. |
| <img src="kmeans_err.png" width=100% height=100%> | The accuracy of the search  can also waver due to possible chances of excluding potential nearest vectors not in the nearest k-cluster. Because errors occur by chance, numbers will fluctuate but are expected to increase as the number of clusters increase. Searches with fewer clusters have less errors than those with more. Still, the percentage of error are less than 1%.|


## Abstract
Modern devices allow users to search for certain files, but solely utilize the names of the files, instead of the contents stored within. As humanity progresses technologically, the increasing amounts of data will serve as a barrier to a practical application of the latter task in terms of time and cost. This will necessitate accurate methods for finding information in a timely and effective manner. In this work, we create a system to search for relevant content within texts and images in respect to a user’s queries utilizing NLP that is both productive and accurate when tested on over 300,000 sources of data.

## Prerequisites

**Installation**

Python Modules

```
pip install torch
pip install transformers
pip install tabulate
pip install numpy
pip install tqdm
pip install argparse
pip install sklearn
pip install PyPDF2
pip install pytesseract
pip install tqdm
pip install pdf2image
```

Tesseract OCR

```
sudo apt install tesseract-ocr
```

SpaCy Profile
```
python3 -m spacy download en_core_web_sm
```

## Usage

You will need pdf files to search through. You may download these sample [files](https://www.dropbox.com/sh/4gedwm2sc7ylsxf/AAB798H6sdVW4n9iV5TZWF5Qa?dl=0) or use your own. Store these source files in a directory. By default, this directory would be `/home/{user}/daviesearch/` on a Linux machine.

Then run `doc_retrieval.py`, which uses the following arguments:

- `--source`: path of folder with source files (default: /home/{user}/daviesearch/)
- `--data`: path of folder to store program data (default: /home/{user}/daviesearch_data/)
- `--query`: query to search for (optional: if not provided, will prompt for one while running)
- `--noscan`: disable scanning for new files in source directory to extract text and generate embeddings for, which is done by default (default: False)
- `--mode`: enable/disable optimization (default: base,0 - if you would like to enable optimization, use kmeans,4)

By default, each time you run the program, it will extract text and generate embeddings for any new sources (which will include everything when running this program for the first time). This may take a significant amount of time for large datasets.

## Future Work Ideas

- Multithreading
- Background program to automatically extract text and generate embeddings
