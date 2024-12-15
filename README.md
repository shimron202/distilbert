# distilbert
distilbert pre trained model based on imdb review movies 

README: DistilBERT Model Inference
Project Overview
This project provides a pre-trained DistilBERT model in the folder my_distilbert_model and a Google Colab notebook (DistilBERT.ipynb) to run inference and evaluation. Follow the steps below to upload the model folder, execute the notebook, and test the model.

Requirements
Google Colab account.
The folder my_distilbert_model (containing the pre-trained model).
The notebook file DistilBERT.ipynb.
Steps to Run the Project
1. Open Google Colab
Go to Google Colab.

Create a new notebook or open an existing one.
2. Upload the Notebook
Upload the file DistilBERT.ipynb:

Click the folder icon on the left-hand side in Colab.
Click the upload button and select DistilBERT.ipynb.
Open the uploaded notebook.

3. Upload the Model Folder
Upload the my_distilbert_model folder directly:

Go to the Files panel on the left-hand side in Google Colab.
Drag and drop the entire my_distilbert_model folder into the Colab workspace.
After uploading, ensure the folder structure looks like this:
/content/my_distilbert_model/
    ├── config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
    
4. Follow the Notebook Instructions
Locate the section in the notebook titled "Test start from here".
Follow the instructions provided in the notebook.
This section will automatically load the uploaded my_distilbert_model folder for testing.


5. Run the Notebook
Execute the cells in the notebook sequentially.
Monitor the outputs:
You will see a classification report and other evaluation metrics.

6. Expected Results
The notebook will produce:
A classification report with Precision, Recall, and F1-Score.
Outputs of the model predictions on a test dataset.


Final Notes
Ensure the my_distilbert_model folder is uploaded directly and visible in the Colab Files panel.
Run all code cells step-by-step as indicated in the notebook.


Contact
For any issues or clarifications, please reach out to the project author. 🚀
