# apple_tweets_nlp
NLP on customer tweets to Apple Support to categorize issues and predict the product group based on initiating tweets

This repo contains:
- **apple_tweets_preprocessing.py**: data preprocessing
- **apple_tweets_modeling.ipynb**: topic modeling using NMF and CorEx
- **data**: pickled files
- **apple_tweets_nlp_slides.pdf**: pdf of project presentation slides

**Note 1:** The proprocessing python script is designed to run where mongodb has already been setup and pre-stored with the tweet data (sourced from Kaggle: kaggle.com/thoughtvector/customer-support-on-twitter)  
**Note 2:** Some data files were excluded from the data folder due to github's size limitation
