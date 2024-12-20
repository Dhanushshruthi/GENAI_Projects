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
**4.Question Answering Chatbot using Dense Retrieval and BERT**

This project implements a Question Answering (QA) chatbot using a combination of Dense Retrieval with Sentence Transformers and BERT for context-based answer generation. The system is designed to take a PDF file as input, extract its content, and allow users to interactively ask questions based on the information within the document.

**Features**

**PDF Content Parsing:** Automatically extracts text from a PDF document using the pdfplumber library.

**Dense Retrieval:** Uses Sentence Transformers (all-mpnet-base-v2) to find the most relevant passages for a given question.

**BERT-based Question Answering:** Employs bert-large-uncased-whole-word-masking-finetuned-squad to generate precise answers from the retrieved passages.

**Interactive Chatbot:** Enables users to ask questions dynamically in a conversational format.

**Acknowledgments**

Transformers Library by Hugging Face.

Sentence Transformers for dense embedding-based retrieval.

pdfplumber for PDF text extraction
```bash
!pip install pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
```
**5.Text-to-Speech Generation**

**Overview**

This project focuses on generating natural-sounding speech from text using pre-trained models and the **Coqui TTS library**. The system has been implemented to support two functionalities:

Text-to-speech generation for a single line or paragraph of text.

Text-to-speech generation for the entire content of a PDF file.

The generated speech is saved as a .wav file, which can be downloaded or played directly.

**Implementation Details**

**Pre-Trained Model:**

The project uses the Coqui TTS library with a pre-trained model (tts_models/en/ljspeech/tacotron2-DDC) to generate high-quality speech.

**Single Text-to-Speech:**

The input text is passed to the synthesizer to produce a .wav audio file.

**PDF-to-Speech:**

Text is extracted from the PDF file using the PyMuPDF library.

The extracted content is passed to the synthesizer, generating a .wav file that narrates the entire document.

**File Handling:**

The generated audio file is saved locally as output.wav.

If executed in environments like Google Colab, the file can be downloaded for local playback.

**Required libraries**

**Coqui TTS:** For text-to-speech synthesis

**PyMuPDF:** For extracting text from PDFs

**SoundFile:** For handling audio files

**playsound:** For playing audio files (optional)
```bash
pip install TTS PyMuPDF soundfile playsound
```

**6.Multi-Lingual Sentiment Analysis**

**OverView**

**Pre-trained Model:** The model used (nlptown/bert-base-multilingual-uncased-sentiment) is a pre-trained BERT-based model fine-tuned for sentiment analysis across multiple languages.

**Sentiment Classification:** The sentiment analysis pipeline uses the model to predict a "star rating" (1-5 stars), which is then mapped to one of three categories: "Happy", "Sad", or "Neutral".

**Score Classification:** Based on the confidence score returned by the model, the text is classified into one of the sentiment categories:

If the score is greater than 0.75, the sentiment is classified as "Happy".

If the score is between 0.55 and 0.75, the sentiment is classified as "Sad".

If the score is less than 0.55, the sentiment is classified as "Neutral".

**Requirements**

**transformers:** Provides access to pre-trained models for various NLP tasks, including sentiment analysis.

**torch:** Required for running deep learning models, including those provided by the transformers library.
```bash
pip install transformers torch
```


**7.Few Shot Learning**

**Overview**

This project implements Few-Shot Learning using a Siamese Network. The model is trained to determine whether  input images belong to the same class or different classes based on their similarity. It utilizes a contrastive loss function to minimize the distance between similar images and maximize the distance between dissimilar images.

**How It Works**

**Siamese Network:** The network consists of two identical CNN branches that process the input images. The outputs are then compared using a distance metric (Euclidean distance in this case).

**Contrastive Loss:**  The model is trained using a contrastive loss function that encourages the network to minimize the distance between pairs of images that belong to the same class and maximize the distance for pairs of images that belong to different classes.

**Dataset:** The dataset consists of image pairs that are either from the same class or different classes. The model learns to classify pairs of images by comparing their feature vectors.

**Libraries:**

**1.PyTorch:** For deep learning model creation and training.
```bash
pip install torch
```
**2.torchvision:** For pre-trained models and image transformations.
```bash
pip install torchvision
```
**3.PIL (Pillow):** For image processing and manipulation.
```bash
pip install pillow
```
**4.NumPy:** For handling numerical operations.
```bash
pip install numpy
```

**8.Neural Network Quantization**

**Overview**

This repository demonstrates a simple approach for training a neural network on the MNIST dataset using PyTorch, followed by applying dynamic quantization to reduce the model size and improve inference performance. The code includes training a neural network to classify MNIST digits, performing model quantization, and using the quantized model to make predictions on test images.

**Key Steps:**

**Training a Neural Network on MNIST:** A simple feed-forward neural network (SimpleNN) is trained on the MNIST dataset for digit classification.

**Dynamic Quantization:** After training, dynamic quantization is applied to the model to reduce the precision of the model weights, resulting in a smaller and faster model for inference.

**Prediction and Visualization:** The quantized model is then used to predict labels for test images, which are displayed along with their true labels.

**Requirements**

**PyTorch:** For deep learning model training, quantization, and inference.

**TorchVision:** For loading the MNIST dataset.

**Matplotlib:** For displaying the test images and results.
```bash
pip install torch torchvision matplotlib
```
**How It Works**

**MNIST Dataset:** The MNIST dataset consists of 28x28 grayscale images of digits (0-9), split into training and test sets. Images are normalized to a range of [0, 1].

**Model Definition:** A simple feed-forward neural network is used with:

  **Input Layer:** Flattened 28x28 image 

  **Hidden Layer:** 512 units with ReLU activation

  **Output Layer:** 10 units (for digits 0-9)

**Training:** The model is trained for 3 epochs using CrossEntropy loss and Adam optimizer. Accuracy and loss are displayed for each epoch.

**Quantization:** Dynamic quantization is applied to the model, reducing the size by converting linear layers to 8-bit integers.
```bash
torch.qint8
```
**Prediction:** After quantization, the model predicts labels for test images based on a user-selected index, displaying the true and predicted labels.

**Saving the Quantized Model:** The quantized model is saved to quantized_model.pt for later use.

**9.Text Summarization Using BART**

**Overview**

This project demonstrates how to use the facebook/bart-large-cnn model for text summarization, It leverages the DialogSum dataset, a conversational dataset, to generate summaries for dialogue-based text using the Hugging Face transformers library. The process is performed in a simple pipeline that extracts text from the dataset, applies summarization, and outputs the results.

**How to Use**

**Install Required Libraries:** First, install the necessary libraries using pip. Open your terminal and run
```bash
!pip install datasets transformers torch
```
**Load the Dataset:** I used the DialogSum dataset, which can be loaded using the datasets library. The dataset contains conversations and corresponding summaries. The following code loads the training set from the dataset.

**Summarization Pipeline:** Use the Hugging Face transformers pipeline to perform text summarization on the dialogue. The pipeline simplifies the process of applying the model to the input text.

**Requirements**

**transformers library:** A library by Hugging Face that provides pre-trained models for natural language processing tasks, including summarization.

**datasets library:** A library by Hugging Face for easy access and manipulation of datasets, such as the DialogSum dataset used in this project.

**torch (PyTorch):** A deep learning framework needed for running transformer models like BART.

**pandas:** A library for handling and processing datasets in tabular form, used when manipulating the dataset.

**10.Text-generation using GPT2**

**Overview**

The project leverages the GPT-2 large model to perform text generation based on a provided input sentence. The generated text is extended from the input while adhering to constraints like avoiding repetition. 

Tokenization of input text.

Encoding and decoding of text into numerical formats suitable for GPT-2.

Text generation using beam search for quality results.

**Key Features**

**Pretrained Model:** Uses gpt2-large, a larger variant of GPT-2 for improved performance.

**Beam Search:** Ensures quality generation with the num_beams parameter.

**Repetition Control:** Avoids repeating n-grams with no_repeat_ngram_size.

**Notes**

The script uses the eos_token_id as the padding token for compatibility with GPT-2.

The maximum length of the generated text can be adjusted with the max_length parameter.

Ensure your environment has enough resources to load the gpt2-large model.

**Requirements:**

**Transformers library:** Provides pre-trained models and tools for text generation and NLP tasks.

**Torch library:** Essential for handling tensors and running deep learning models efficiently.
```bash
pip install transformers datasets 

```












