# CMPT-733-Big-Data-2
Course Project for CMPT 733 (Programming for Big Data 2)


Data Cleaning and Integration:
The folder contains:
-	Script for preprocessing the Wikipedia, SOCC and Facebook data. (Data_Cleaning.py)
-	Script for merging the Wikipedia and SOCC datasets (Merging_Datasets.py)
-	Script for merging the files scraped from facebook to make one final CSV for each category (merge_csvs.py)

Models:
The Folder contains:
-	RNN folder with an implementation of RNN model
-	TFIDF implementation of multiple scikit learn models:
	-	tfidf_scikit.py
	-	tfidf_saved_model.py

-	Scripts for Spark implementation of doc2vec model (doc2vec2csv.py, doc2vec_spark_cluster.py)
-	Script for Scikit Learn's implementation of doc2vec (doc2vec.ipynb)


Analysis:
The Folder contains:
-	topic_modeling.py:	Script for topic modeling using LDA and visualizing the results.
-	Slangs List Big Data.csv:	List for detecting the type of toxicities. Contains sexist, racist and homophobic words
-	categorize_toxicity.py:		Script for finding the different types of toxicities 
-	Confidence_Interval.ipynb:	Script for calculating the confidence interval and generalizing our methods.
-	Trends_Analysis.ipynb:	Script for plotting the line graph for CNN.

Visualizations:
The Folder contains the images/charts we produced in our project.

Data:
The Folder contains:
-	The CSVs we produced and used throughout our project
-	Facebook Scraping.ipynb in Data/Facebook Data is the script for scraping data from facebook.


How to run the code for RNN (Final Model):

Run the RNN.ipynb file in the Models/RNN folder. It will create a predictions csv, the predicions file should be given to the categorize_toxicity.py file in Analysis folder. The output of that script would be a statistics.csv file which would be given to Confidence_Interval.ipynb file in the Analysis folder. The output of that file will be used to generate the Tableau visualizations. In order to create the cnn line chart, run Trends_Analysis.ipynb file with the cnn_Predictions.csv file in Data/Facebook Data/predictions/Final_Predictions folder. 

For running the python file:
python3 file_name

