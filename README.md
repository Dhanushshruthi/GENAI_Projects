**1.TextSentimentClassifier**

**Overview of the modal:**
The **cardiffnlp/twitter-roberta-base-sentiment model** is a pre-trained transformer model based on RoBERTa, fine-tuned specifically for sentiment analysis on Twitter data. It is designed to classify text into three sentiment categories:

Positive

Negative

Neutral

This model has been trained on a large dataset of tweets, making it highly effective for analyzing social media text, which often contains informal language, slang, and emoticons.

**cardiffnlp/twitter-roberta-base-sentiment Model**

**Type:** RoBERTa-based transformer model.

**Task:** Sentiment analysis (Positive, Negative, Neutral).

**Data:** Fine-tuned on Twitter data.

**Output:** Classifies text into 3 sentiment labels: Negative, Neutral, Positive.

**Application:** Suitable for analyzing informal, social media text like tweets.


**Required Libraries:**
```bash
pip install transformers
pip install torch
pip install matplotlib
pip install pandas
```
**2.Chatbot with Retrieval-Augmented Generation (RAG)**

**Overview**

**Objective:** A PDF Question Answering (QA) Chatbot that extracts information from a PDF and answers user queries.

**Core Technologies:**

BERT: For question answering based on the extracted context.

FAISS: For efficient similarity search to find relevant document chunks.

**Process:**

1.Extracts text from the PDF document.

2.Splits the text into smaller chunks.

3.Generates embeddings for each chunk using Sentence-BERT.

4.Stores embeddings in a FAISS index for fast similarity-based retrieval.

5.For each query, retrieves the most relevant chunks and uses the fine-tuned BERT model to generate an answer.

**Use Case:** Ideal for creating conversational agents that interact with large documents, such as manuals, reports, or eBooks, in a Q&A format.

**Required Libraries**

**PyMuPDF (fitz):** Used for extracting text from PDF documents.

**Sentence-Transformers:** Provides pre-trained models to generate embeddings for document chunks.

**FAISS:** Efficient similarity search library used to index and retrieve relevant chunks from document embeddings.

**Transformers (Hugging Face):** Used for loading and using pre-trained BERT models fine-tuned for Question Answering.

```bash
pip install fitz
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install transformers
```
**3.Image Captioning with BLIP Model**

This repository demonstrates image caption generation using the BLIP (Bootstrapped Language-Image Pretraining) model. The project consists of two key parts:

Training on the CIFAR-10 Dataset

Generating Captions for a Custom Dataset

**Requirements**
```bash
pip install transformers pillow
pip install torch torchvision
```

**1. Using CIFAR-10 Dataset**

The CIFAR-10 dataset was loaded, preprocessed, and fed into the BLIP model for caption generation. The steps included:

Resizing images to 128x128 for model compatibility.

Using the Salesforce/blip-image-captioning-base pre-trained model for captioning.

Displaying the generated captions along with the images.
```bash
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
```

**2. Using a Custom Dataset**

The model was extended to generate captions for images in a custom dataset. This involved:

Loading images from a specified folder.

Processing each image using the BLIP processor.

Generating and displaying captions for the images.
```bash
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
import os
```