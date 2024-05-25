Gpt-neox needs to pre-tokenize the data into two memory mapped arrays: an .idx file and a .bin file.

Quoting gpt-neox instructions to pre-tokenize custom data: To prepare your own dataset for training with custom data, format it as one large jsonl-formatted file with each item in the list of dictionaries being a separate document. The document text should be grouped under one JSON key, i.e "text". Any auxiliary data stored in other fields will not be used.

As such, we need to convert our datasets to a single .jsonl file where each column has the json key "text" and json value a text (string) document.

These are the scripts I used to pre-tokenize datasets.

1. download_dataset.py --> Download dataset of your choice, I have added many Yihong reccomended textbook datasets in notion's word. The dataset will be downloaded to the cluster's cache.
2. verify_arrow_file_format.py --> huggingface datasets are downloaded as .arrow files. It is unclear what file format they actually are as they can be Parquet, Feather or Arrow IPC Streaming format with such file extension (from what I could tell). This file verifies what file format the dataset you downloaded is in. In my own experience, they have always been Arrow IPC Streaming format. Warning: in my own experience, if you download the dataset locally, the file format appears to be parquet but if you download it on the cluster it's Arrow IPC Streaming format.
3. verify_data_columns.py --> Allows to view available columns on dataset. This information should also be on huggingface, however, some of the datasets I encountered did not have such info on huggingface. Once downloaded and file format identified, decide what column must be extracted to the jsonl file gpt-neox requires. It should be the column that actually contains the string of text for our model to be pre-trained on. 
4. arrow_to_jsonl.py --> Converts arrow files to single jsonl file following gpt-neox requirements. You must specify column to extract. Tweak to your needs. This script has only been battle-tested for Arrow IPC Streaming format files. Unsure if it works for Parquet or Feather.
5. pre_tokenize_HFTokenizer.sh --> Pre-tokenizes jsonl file. Tweak to your needs.
