# JupyterLab Workspace Backup

A comprehensive archive of Jupyter notebooks, machine learning models, computer vision scripts, and AI projects developed locally.

---

## Project Index

| Project Directory | Category / Topic | Primary Notebooks & Scripts | Key Assets / Notes |
| :--- | :--- | :--- | :--- |
| **`ContextAwareVideoCaptioning`** | Vision & LLM Captioning | `context_aware_video.ipynb`, `main.py`, `test_qwen.py` | Web UI (`index.html`), video database JSON |
| **`DecisionTrees`** | Supervised ML | `DecisionTree.ipynb`, `StressLevel.ipynb`, `avengers_doomsday_cyberwar_notebook.ipynb` | `StressLevelDataset.csv`, `Titanic.csv` |
| **`Flask`** | Web Application & ML Deployment | `FlaskBasic/app.py`, `Digit Classifier/app.py`, `MNIST_CNN.ipynb` | Trained models (`mnist.h5`, `mnist.keras`), HTML UI templates |
| **`gadgets360-AI`** | LLM Chatbot Backend | `backend/main.py`, `backend/diagnose_reviews.py` | Fast API/Flask backend for tech gadget reviews (also hosted at [`Gadgets360-Ai-Chatbot`](https://github.com/Chinmay-Kapila/Gadgets360-Ai-Chatbot)) |
| **`GANs`** | Generative Models | `DCGANs.ipynb`, `Vanilla_GANs.ipynb` | Visual generated sample outputs across training epochs |
| **`GenAI`** | LLM Pipelines & RAG | `Data Ingestion/`, `Data Transformer/`, `Embeddings/`, `Vector Store/`, `Ollama/` | FAISS, Chroma, LangChain, Ollama, HuggingFace embeddings |
| **`Image Captioning`** | Deep Learning Vision & Language | `Image Captioning Project.ipynb` | Feature vectors (`encoded_train_features.pkl`, `encoded_test_features.pkl`), descriptions |
| **`KMeans`** | Unsupervised ML | `KMeans.ipynb`, `Dominant Color Extraction for Image Segmentation.ipynb` | Image segmentation demos (`elephant.jpg`, `example.png`) |
| **`Markov`** | NLP & Sequence Modeling | `Markov.ipynb`, `Word2Vec Basics - Master.ipynb` | Text corpora (`bollywood.txt`, `english_speech_2.txt`) |
| **`Model Wars`** | ML Competition Classification | `avengers_doomsday.ipynb` | `feature_map.csv`, `label_map.csv`, `test.csv` (train dataset stored locally) |
| **`NeuralNetwork`** | Deep Learning Fundamentals | `NN.ipynb` | Feedforward neural network implementation |
| **`NLP, Naive Bayes`** | Text Mining & Classification | `NaiveBayes.ipynb`, `NLP.ipynb` | `emails.csv` (spam detection), `mushrooms.csv` |
| **`OpenCV`** | Computer Vision | `FaceDetection.py`, `FaceRecog.py`, `Face_Data.py`, `CV_Video_Read.py` | Haar cascade XML, face embedding `.npy` files |
| **`pca`** | Dimensionality Reduction | `PCA.ipynb`, `CancerPrediction.ipynb` | Principal component analysis & breast cancer classification |
| **`Reinforcement Learning`** | Reinforcement Learning | `RL_Agent.ipynb`, `RL_Gym.ipynb` | Q-learning agent and OpenAI Gym environment |
| **`RNN`** | Sequence Models & NLP | `RNN.ipynb`, `LSTM/Emoji_Prediction.ipynb` | LSTM emoji sentiment classification datasets |
| **`Selenium`** | Web Scraping & Automation | `SearchYahoo.py`, `abcdefg.py`, `test.py` | Selenium WebDriver automation scripts |
| **`SubtitleAnalyzer`** | NLP & Sentiment Analysis | `subtitle_analyzer.ipynb` | Evaluation curves and confusion matrix plots (also hosted at [`Subtitle-Genre-Analyzer`](https://github.com/Chinmay-Kapila/Subtitle-Genre-Analyzer)) |
| **`TensorFlow`** | Deep Learning & CNNs | `HindiCharacter.ipynb`, `TF_CNN_Cifar10.ipynb`, `TF_CNN_MNIST.ipynb`, `TF_CNN_TransferLearning.ipynb` | Computer vision classification architectures |
| **`Untitled.ipynb`** | Search Algorithms | `Untitled.ipynb` (root) | Best-First Search and heuristic graph traversal |

---

## Notes on Architecture & Exclusions
- **Environment & Caches**: Python virtual environments (`.venv`, `miniconda3`), temporary caches (`__pycache__`), and checkpoint files (`.ipynb_checkpoints`) are excluded.
- **Large Assets**: Files exceeding GitHub's 100 MB file size limit (such as `Model Wars/train.csv` at 220.6 MB) are preserved in local storage and excluded from remote Git commits unless Git LFS is configured.
- **Standalone Repositories**: Projects with dedicated GitHub repositories (`gadgets360-AI` and `SubtitleAnalyzer`) have their source files tracked in this master archive with their original standalone git repositories untouched.
