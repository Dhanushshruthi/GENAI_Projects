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


