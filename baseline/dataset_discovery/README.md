This subdirectory consists of three baseline discovery models.
'data_reader.py' is a data reader file that is used in the baseline models.
The scores printed when running the scripts are based on the 0 and 1 labels predicted, not the exact start and end index match. For example, true positive means this token is predicted as part of a dataset mention while it is indeed part of a dataset mention.
The scripts will also create a file in the outputs folder to store the start and end index predicted, which can be used for exact match evaluation. 
The format of output files: start index, end index, dataset_id, publication_id.