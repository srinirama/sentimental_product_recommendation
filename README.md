# Capstone -Sentiment Based Product Recommendation System

### Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

### Live Version

* PythonAnywhere PAAS (Application is Live): http://sramanujam.pythonanywhere.com/

### ScreenShots
* Input Screen
 ![input-screen.jpg](.%2Fimages%2Finput-screen.jpg)
* Top 5 Product Recommendation Screen
 ![recommendation-screen.jpg](.%2Fimages%2Frecommendation-screen.jpg)

### Technology Stack

* Python 3.9.7
* scikit-learn 1.0.2
* xgboost 1.5.1
* numpy 1.22.0
* nltk 3.6.7
* pandas 1.3.5
* Flask 2.0.2
* Bootstrap CDN 5.1.3

### Solution Approach

* Dataset and attribute descriptions can be found in the "dataset" folder.
* Data cleaning, visualization, and text preprocessing (NLP) techniques are applied to the dataset. The TF-IDF Vectorizer is used to convert the textual data (review_title + review_text) into vectors, which measure the relative importance of words in comparison to other documents.
* The dataset suffers from a class imbalance issue, and the SMOTE oversampling technique is applied before applying the machine learning models.
* Machine learning classification models such as Logistic Regression, Naive Bayes, and Tree Algorithms (Decision Tree, Random Forest, XGBoost) are applied on the vectorized data and the target column (user_sentiment). The objective of this ML model is to classify sentiment as positive (1) or negative (0). The best model is selected based on various ML classification metrics such as Accuracy, Precision, Recall, F1 Score, and AUC. XGBoost is chosen as the better model based on the evaluation metrics.
* A collaborative filtering recommender system is created based on user-user and item-item approaches. RMSE (Root Mean Square Error) is used as the evaluation metric.
* The code for sentiment classification and recommender systems can be found in the "SentimentBasedProductRecommendation.ipynb" Jupyter notebook.
* The top 20 products are filtered using the improved recommender system, and for each product, the user_sentiment is predicted for all the reviews. The top 5 products with higher positive user sentiment are filtered out using the "model.py" file.
* The machine learning models are saved in pickle files under the "pickle" folder. The Flask API (app.py) is used to interface and test the machine learning models. Bootstrap and Flask Jinja templates (templates\index.html) are used for setting up the user interface. No additional custom styles are used.
* The end-to-end application is deployed on PythonAnywhere ( a PAAS application).




