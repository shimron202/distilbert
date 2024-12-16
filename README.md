# **DistilBERT**

**DistilBERT pre-trained model based on IMDb review movies**

---

## **README: DistilBERT Model Training and Inference**

This project provides two Python files:  
- `distilbert.py`: Used to **train** the DistilBERT model locally.  
- `run.py`: Used to **test** the trained model for inference and evaluation.

Follow the steps below to set up the environment, install dependencies, train the model, and run the test.

---

## **Requirements**

- Python 3.8 or higher  
- GPU support (optional but recommended for faster training)  
- Required libraries:  
   - `transformers`, `datasets`, `torch`, `scikit-learn`  

---

## **download the pre trained model**
if you dont wish to train the model yourself download the pre trained model from this link:https://drive.google.com/drive/folders/19ky95SqTLRNCpr7zmoAXT9locCWGcxwi?usp=sharing
and skip to phase 4.

## **1. Install Dependencies**

Before running the project, install all required libraries using `pip`.

Run the following command in your terminal:

```bash
pip install transformers datasets torch scikit-learn
```

---

---
## **2. Set Up the Files**

Download the two files from this repository:

- `distilbert.py` (Training File)  
- `run.py` (Test/Inference File)  

Ensure both files are in the same folder on your local machine:

```bash
project_folder/
│
├── distilbert.py   # Training file
├── run.py          # Test file
```
---

## **3. Train the Model**

1. Run the `distilbert.py` file to train the model.

2. Once training is complete, a new folder named `my_distilbert_model` will be created in the same directory.

**Folder Structure After Training:**

```bash
project_folder/
│
├── distilbert.py
├── run.py
├── my_distilbert_model/   # Folder created after training
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
```
## **4. Test the Model**

1. Once the training is complete and the `my_distilbert_model` folder is available:

   Run the `run.py` file to test the model:

   ```bash
   python run.py
   ```

---

## **Final Notes**

- Make sure that the `distilbert.py` and `run.py` files are in the same folder.  
- Confirm that the `my_distilbert_model` folder is generated successfully after training.  
- Run all steps sequentially to avoid errors.


---


## **Contact**

For any issues or clarifications, please reach out to the project author. 🚀

