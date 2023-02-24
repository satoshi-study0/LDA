# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:54:42 2023

@author: Takada Satoshi
"""

import pandas as pd
import MeCab
import gensim
import numpy as np
import matplotlib.pyplot as plt

livedoor = pd.read_csv("livedoornews.csv")
livedoor.head()

# 形態素解析
def parse(tweet_temp):
    t = MeCab.Tagger()
    temp1 = t.parse(tweet_temp)
    temp2 = temp1.split("\n")
    t_list = []
    for keitaiso in temp2:
        if keitaiso not in ["EOS",""]:
            word,hinshi = keitaiso.split("\t")
            t_temp = [word]+hinshi.split(",")
            if len(t_temp) != 10:
                t_temp += ["*"]*(10 - len(t_temp))
            t_list.append(t_temp)

    return t_list

def parse_to_df(tweet_temp):
    return pd.DataFrame(parse(tweet_temp),
                        columns=["単語","品詞","品詞細分類1",
                                 "品詞細分類2","品詞細分類3",
                                 "活用型","活用形","原形","読み","発音"])


# 単語をBag-of-Words形式で保存し、LDAを使いやすい形に変形。今回は、一般名詞と固有名詞のみを抽出
def make_lda_docs(texts):
    docs = []
    for text in texts:
        df = parse_to_df(text)
        extract_df = df[(df["品詞"]+"/"+df["品詞細分類1"]).isin(["名詞/一般","名詞/固有名詞"])]
        extract_df = extract_df[extract_df["原形"]!="*"]
        doc = []
        for genkei in extract_df["原形"]:
            doc.append(genkei)
        docs.append(doc)
    return docs

texts = livedoor["body"].values
docs = make_lda_docs(texts)
dictionary = gensim.corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]


n_cluster = 6 #クラスターの数、何回も変更してみて決めよう
lda = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n_cluster, 
                minimum_probability=0.001,
                passes=20, 
                update_every=0, 
                chunksize=10000,
                random_state=1
                )

corpus_lda = lda[corpus]
arr = gensim.matutils.corpus2dense(
        corpus_lda,
        num_terms=n_cluster
        ).T

# トピック-単語分布の可視化
lists = []
for i in range(n_cluster):
    temp_df = pd.DataFrame(lda.show_topic(i),columns=["word","score"])
    temp_df["topic"] = i
    lists.append(temp_df)
topic_word_df = pd.concat(lists,ignore_index=True)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
for i,gdf in topic_word_df.groupby("topic"):
    gdf.set_index("word")["score"].sort_values().plot.barh(
        ax=axes[i//3, i%3],
        title="topic {} のトピック-単語分布".format(i),
        color="blue")
    
# テーブルで順位を表示
topic_word_df["rank"] = topic_word_df.groupby("topic")["score"].rank()
topic_word_df.pivot(index='topic', columns='rank', values='word')


# 文書-トピック分布の可視化
livedoor_predict = livedoor.copy()
# topicの付与
livedoor_predict["pred_topic"] = np.argmax(arr,axis=1)
livedoor_predict["score"] = np.max(arr,axis=1)
cross = pd.crosstab(livedoor_predict["media"],livedoor_predict["pred_topic"])
# トピックの文書割合の可視化
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(cross)):
    ax.barh(y=cross.columns, width = cross.iloc[i].values[::-1], left=cross.iloc[:i].sum()[::-1].values,tick_label=cross.columns[::-1])
ax.set(xlabel='個数', ylabel='トピック')
ax.legend(cross.index)
plt.show()