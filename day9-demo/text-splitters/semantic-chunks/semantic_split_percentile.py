# pip install --quiet langchain_experimental langchain_openai
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 这是一个长文档，我们可以将其拆分。
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90
)
docs = text_splitter.create_documents([knowledge])
print(docs[0].page_content)
print(docs[1].page_content)
print(len(docs))

# 50%
# ﻿I am honored to be with you today at your commencement from one of the finest universities in the world. I never graduated from college.
# Truth be told, this is the closest I've ever gotten to a college graduation. Today I want to tell you three stories from my life.
# 71

# 90%
# ﻿I am honored to be with you today at your commencement from one of the finest universities in the world. I never graduated from college. Truth be told, this is the closest I've ever gotten to a college graduation. Today I want to tell you three stories from my life. That's it. No big deal.
# Just three stories. 我今天很荣幸能和你们一起参加毕业典礼，斯坦福大学是世界上最好的大学之一。我从来没有从大学中毕业。说实话,今天也许是在我的生命中离大学毕业最近的一天了。今天我想向你们讲述我生活中的三个故事。不是什么大不了的事情,只是三个故事而已。
#
# The first story is about connecting the dots. 第一个故事是关于如何把生命中的点点滴滴串连起来。
#
# I dropped out of Reed College after the first 6 months, but then stayed around as a drop-in for another 18 months or so before I really quit. So why did I drop out? 我在Reed大学读了六个月之后就退学了，但是在十八个月以后——我真正的作出退学决定之前，我还经常去学校。我为什么要退学呢？
#
# It started before I was born. My biological mother was a young, unwed college graduate student, and she decided to put me up for adoption. She felt very strongly that I should be adopted by college graduates, so everything was all set for me to be adopted at birth by a lawyer and his wife. Except that when I popped out they decided at the last minute that they really wanted a girl. So my parents, who were on a waiting list, got a call in the middle of the night asking: "We have an unexpected baby boy; do you want him?" They said: "Of course." My biological mother later found out that my mother had never graduated from college and that my father had never graduated from high school. She refused to sign the final adoption papers. She only relented a few months later when my parents promised that I would someday go to college. 故事从我出生的时候讲起。我的亲生母亲是一个年轻的、没有结婚的大学毕业生。她决定让别人收养我，她十分想让我被大学毕业生收养。所以在我出生的时候，她已经做好了一切的准备工作，能使得我被一个律师和他的妻子所收养。但是她没有料到，当我出生之后，律师夫妇突然决定他们想要一个女孩。所以我的生养父母（他们还在我亲生父母的观察名单上）突然在半夜接到了一个电话：“我们现在这儿有一个不小心生出来的男婴,你们想要他吗？”他们回答道：“当然！”但是我亲生母亲随后发现，我的养母从来没有上过大学，我的父亲甚至从没有读过高中。她拒绝签这个收养合同。只是在几个月以后，我的父母答应她一定要让我上大学，那个时候她才同意。
#
# And 17 years later I did go to college. But I naively chose a college that was almost as expensive as Stanford, and all of my working-class parents' savings were being spent on my college tuition.
# 15