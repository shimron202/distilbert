# distilbert
distilbert pre trained model based on imdb review movies 

# **DistilBERT**

**DistilBERT pre-trained model based on IMDb review movies**

---

## **README: DistilBERT Model Inference**

This project provides a pre-trained **DistilBERT** model in the folder `my_distilbert_model` and a Google Colab notebook `DistilBERT.ipynb` to run inference and evaluation. Follow the steps below to upload the model folder, execute the notebook, and test the model.

---

## **Requirements**

- **Google Colab** account.
- Folder: `my_distilbert_model` (contains the pre-trained DistilBERT model).
- Notebook file: `DistilBERT.ipynb`.

---

## **Steps to Run the Project**

### **1. Open Google Colab**
- Go to [Google Colab](https://colab.research.google.com/).
- Create a new notebook or open an existing one.

---

### **2. Upload the Notebook**
- Click the **folder icon** on the left-hand side in Colab.  
- Click the **Upload** button and select `DistilBERT.ipynb`.  
- Open the uploaded notebook.

---

### **3. Upload the Model Folder**
- Go to the **Files** panel on the left-hand side in Google Colab.  
- **Drag and drop** the entire folder `my_distilbert_model` into the Colab workspace.  

### **Folder Structure**

Your uploaded folder structure should look like this:

```plaintext
/content/my_distilbert_model/
│
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt





---

### **4. Follow the Notebook Instructions**
- Locate the section in the notebook titled **"Test start from here"**.  
- Follow the instructions to load and test the uploaded `my_distilbert_model` folder.

---

### **5. Run the Notebook**
- Execute the cells in the notebook **sequentially**.
- Monitor the outputs. You will see:
  - **Classification report**.
  - **Model predictions** on the test dataset.

---

### **6. Expected Results**
The notebook will produce:
- A **classification report** summarizing:
   - Precision, Recall, and F1-Score.
- Outputs of the model predictions on a test dataset.

---

## **Final Notes**
- Ensure the `my_distilbert_model` folder is uploaded correctly and visible in the **Colab Files panel**.  
- Run all code cells step-by-step as instructed in the notebook.

---

## **Contact**
For any issues or clarifications, please reach out to the project author. 🚀

