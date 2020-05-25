# display topics and top words for each
def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Takes in a model, feature names and number of top words, returns top words per topic
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '", topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))