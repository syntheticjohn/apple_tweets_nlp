# Text Classification On Customer Tweets To Apple Support
NLP on customer tweets sent to Apple Support to uncover topics over time using NMF (unsupervised topic modeling), and classify tweets as product types based on users' initiating tweets using CorEx with product-focused anchors (semi-supervised topic modeling)

This repo includes:
- **apple_tweets_preprocessing.py**: data preprocessing
- **apple_tweets_modeling.ipynb**: topic modeling using NMF and CorEx
- **data**: pickled files
- **apple_tweets_nlp_slides.pdf**: pdf of project presentation slides

**Note 1:** The proprocessing python script is designed to run where mongodb has already been setup and pre-stored with the tweet data (sourced from: kaggle.com/thoughtvector/customer-support-on-twitter)  
**Note 2:** Some data files were excluded from the data folder due to github's size limitation
