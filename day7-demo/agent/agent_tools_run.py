from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=1)

# pip install langchain
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
# pip install faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "wiki_search",
    "搜索维基百科",
)

model = ChatOpenAI(model="gpt-4")

tools = [search, retriever_tool]

from langchain import hub
# 获取要使用的提示 - 您可以修改这个！
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(model, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#print(agent_executor.invoke({"input": "你好"}))
#print(agent_executor.invoke({"input": "猫的特征"}))
print(agent_executor.invoke({"input": "猫的特征?今天上海天气怎么样?"}))

"""
D:\envs\learn\python.exe D:\Workspace\SecondAIProject\day7-demo\agent\agent_tools_run.py 
USER_AGENT environment variable not set, consider setting it to identify your requests.


> Entering new AgentExecutor chain...

Invoking: `wiki_search` with `{'query': '猫的特征'}`


感官[编辑]
貓的感官適於狩獵。在哺乳類動物中，貓的聽覺、視覺、嗅覺、味覺、觸覺極敏銳。

視覺[编辑]
貓的瞳孔能縮得如線般狹小
貓的瞳孔變化
貓在晝間視覺縱不及人類，夜視能力與追蹤視覺上之活動物件卻相當出色，夜視能力是人類的六倍，雖然綜合色彩計算整體視覺系數則僅及人類的十分之一。貓的眼睛具有微光觀察能力，即使只有微弱月光都可分辨物件，有關光線入貓眼後可放大40至50倍處理，令貓具有夜間活動的能力，即使在黑暗的地下室貓咪依然能活動自如。強光下，貓會將瞳孔縮得如線般狹小，以減少對視網膜的傷害，但視野會因而縮窄。由於貓眼具備高幀與高分辨率視覺，故此電視機極微細之動靜對貓而言亦成逐格跳躍之畫面。[48]
另外，不只是光線會影響貓的瞳孔，感情也會。一般而言貓放鬆的時候瞳孔會縮小，而緊張時會放大。
貓的視網膜背面有一層藍綠色如熒光一般的薄膜（Tapetum Lucidum），可增加在暗處的視力。閃光中，貓眼能呈現各式各樣顔色。如同多數食肉動物，貓眼長在臉上朝正前方，賦予其遼闊的視野，單一貓眼視野為100度，雙眼視野為285度，加上頸旋靈活，總計視野比人類雙眼僅100度靈活得多。不過，貓僅能聚焦其前30厘米至3米之物件，相對人類而言屬大近視。
貓對三原色的辨識力很差，相對人類而言，貓有黃藍色盲且紅綠色辨成灰黃色，與狗相似。關於貓的夜視能力，生物學家發現牛磺酸對貓的視力起了很大作用，貓本身不能合成牛磺酸，必須由外攝取。缺乏牛磺酸，會使視力及夜視能力變差，影響夜間活動。據生物學家稱，貓經常捕鼠，是因老鼠體內含牛磺酸。[49][50]
紫外線可以通過貓的水晶體。
當四周光線微弱，貓會用感覺毛來改善行動力與感知能力。感覺毛主要分布於鼻子兩側、下巴、雙眼上方、兩頰也有數根。感覺毛可感受非常微弱的空氣波動，視野不清時也能協助辨識阻礙位置。鬍鬚尖端與雙耳連成一線，恰是身體能通過障礙的最小範圍，故貓可在黑夜中快速判斷地形能否通過。
貓有第三眼瞼，當貓眼睑張開時，眨眼時第三眼瞼會從旁稍微遮蓋眼睛。若貓生病，或是睡眠，笑著，此眼皮會縮回一部分。若貓長時間嶄露第三眼瞼，表示它的健康有問題。

聽覺[编辑]
貓每隻耳各有32條獨立的肌肉控制耳殼轉動，因此雙耳可單獨朝向不同的音源轉動，使其向獵物移動時仍能對周遭其他音源保持直接接觸。[51] 除了蘇格蘭折耳貓這類基因突變的貓以外，貓極少有狗常見的「垂耳」，多數的貓耳向上直立。當貓忿怒或受驚時，耳朵會貼向後方，並發出咆哮與「嘶」聲。
貓與人類對低頻聲音靈敏度相若。人類中只有極少數的調音師能聽到20 kHz以上的高頻聲音（8.4度的八度音），貓則可達64kHz（10度的八度音），比人類要高1.6個八度音，甚至比狗要高1個八度；但是貓辨別音差須開最少5度，比起人類辨別音差須開最少0.5度來得粗疏。[52][48]

嗅覺[编辑]
家貓的嗅覺較人類靈敏14倍[53]。貓的鼻腔內有2億個嗅覺受器，數量甚至超過某些品種的狗（狗嗅覺細胞約1.25億～2.2億）。

味覺[编辑]
貓早期演化時由於基因突變，失去了甜的味覺，[54]但貓不光能感知酸、苦、鹹味，选择适合自己口味的食物，还能尝出水的味道，这一点是其他动物所不及的。不过总括来说猫的味觉不算完善，相比一般人類平均有9000個味蕾，貓一般平均僅有473個味蕾且不喜好低於室溫之食物。故此，貓辨認食物乃憑嗅覺多於味覺。[48]

觸覺[编辑]
貓在磨蹭時身上會散發出特別的費洛蒙，當這些獨有的費洛蒙留下時，目的就是在宣誓主權，提醒其它貓這是我的，其實這種行為算是一種標記地盤的象徵，會讓牠們有感到安心及安全感。

貓爪[编辑]
貓的爪子尖銳還能有伸縮作用，能夠向外露張開或是往內縮閉藏起來。貓的趾底有脂肪肉墊，因而行走無聲。而按一下這個肉墊，尖銳的勾爪就會伸出來，養貓的人一般使用這種方法來剪爪。大部分貓都有磨爪的習慣。而貓喜好的磨爪物各不相同。家貓可能會選擇沙發、牆壁甚至人類的衣物等作為自己的磨爪物。

感官[编辑]
貓的感官適於狩獵。在哺乳類動物中，貓的聽覺、視覺、嗅覺、味覺、觸覺極敏銳。

各種不同的貓


保护状况


驯养


科学分类 


界：

动物界 Animalia


门：

脊索动物门 Chordata


纲：

哺乳纲 Mammalia


目：

食肉目 Carnivora


科：

猫科 Felidae


属：

貓屬 Felis


种：

貓 F. catus


二名法


Felis catusLinnaeus, 1758


異名



Felis silvestris catus (Linnaeus, 1758)
Felis catus catus (Linnaeus, 1758)
Felis catus domesticus Erxleben, 1777
Felis angorensis Gmelin, 1788
Felis vulgaris Fischer, 1829


一隻英国短毛貓
貓（英語：cat）通常指家貓（學名：Felis catus[1][2][3]，英語：domestic/house cat），是一种家养[4]小型食肉哺乳动物，属小型貓科動物。根據遺傳學及考古學分析，人類養貓的紀錄可追溯至10,000年前的新月沃土地區。古埃及人飼養貓以防止老鼠偷食穀物的紀錄可追溯至公元前1000年前[5]。
現在，貓成為世界上最為廣泛的寵物之一，飼養率僅次於狗，可以治療低落的心情與提供情緒價值等[6]。長期室內飼养的貓平均健康則較長，壽命為12年以上，歷史上最長壽的貓則達38歲[7]，也因此在部分城市中選擇養貓的家戶已經超過了狗[8]。但同時貓是一種掠食動物，大量的貓咪因為弃养接觸外部環境，除了在惡劣的衛生引發疫情散佈，也會造成城市管理問題[9]。其次，这些猫威脅了鳥類、齧齒類的生存。[10]
Invoking: `tavily_search_results_json` with `{'query': '今天上海天气怎么样'}`


[{'url': 'https://tianqi.2345.com/today-58362.htm', 'content': '全国天气 空气质量 历史天气 国际天气 天气资讯 万年历 景点天气 当前： 全国天气> 上海天气预报7天 今天天气 明天天气 一周天气 15天天气 昨天 11/18 阴 11~15° 东北风4级 优 今天 11/19 阴转多云 11~15° 东北风2级 优 明天 11/20 多云转晴 10~16° 北风2级 优 周五 11/22 多云转阴 11~17° 东北风2级 优 周六 11/23 阴转晴 12~16° 东北风3级 优 周日 11/24 阴转小雨 12~17° 东风3级 优 周一 11/25 小雨转中雨 10~18° 西风2级 优 现在 东北风 3级 优 06:00 东北风 3级 优 07:00 东北风 3级 优 08:00 东北风 3级 优 09:00 东北风 3级 优 10:00 东北风 3级 优 11:00 东北风 3级 优 12:00 东北风 3级 优 13:00 东北风 3级 优 更多> PM2.5 PM2.5是指大气中直径小于或等于2.5微米的颗粒物，也称为可入肺颗粒物。被吸入人体后会直接进入支气管，干扰肺部的气体交换，引发包括哮喘、支气管炎和心血管病等方面的疾病。 _7_优 PM10 _21_优 _72_优 _14_优 _7_优 更多> 国际天气 PM2.5 历史天气 全国天气 国际天气 历史天气 景点天气 万年历 天气资讯 空气质量 天气地图 手机网页版 手机APP官网 天气插件 XML地图 更正您的地理位置 意见反馈'}]关于猫的特征：
1. 视觉：猫在晚上具有强大的视觉能力，其夜视能力约为人类的六倍。猫的眼睛具有微光观察能力，即使在微弱的月光下也可以清楚地看到物体。此外，猫对三原色的辨识力很差，它们对黄蓝色盲，并且将红绿色辨别为灰黄色。
2. 听觉：猫的耳朵有32根独立的肌肉来控制耳朵的旋转，使它们可以单独地面向不同的音源转动。此外，猫的听觉对低频声音的敏感度与人类相似，但它们能够听到频率高达64kHz的声音，这比人类高1.6个八度音，并且比狗要高1个八度。
3. 嗅觉：猫的嗅觉比人类灵敏14倍。猫的鼻腔内有2亿个嗅觉受体，数量甚至超过某些品种的狗。
4. 味觉：猫在早期的演化过程中，因为基因突变，失去了甜的味觉，但它们不仅能感知酸，苦，咸味，选择适合自己口味的食物，还能尝出水的味道。
5. 触觉：猫会在蹭人的时候释放特别的信息素。当这些特有的信息素留下时，其目的是在宣誓主权，提醒其他猫这是它们的区域。

关于今天上海的天气，上海的天气预报显示，今天是阴天，露出了一些阳光，温度在11-15℃之间，当前有东北风，风力为3级，空气质量良好。

> Finished chain.
{'input': '猫的特征?今天上海天气怎么样?', 'output': '关于猫的特征：\n1. 视觉：猫在晚上具有强大的视觉能力，其夜视能力约为人类的六倍。猫的眼睛具有微光观察能力，即使在微弱的月光下也可以清楚地看到物体。此外，猫对三原色的辨识力很差，它们对黄蓝色盲，并且将红绿色辨别为灰黄色。\n2. 听觉：猫的耳朵有32根独立的肌肉来控制耳朵的旋转，使它们可以单独地面向不同的音源转动。此外，猫的听觉对低频声音的敏感度与人类相似，但它们能够听到频率高达64kHz的声音，这比人类高1.6个八度音，并且比狗要高1个八度。\n3. 嗅觉：猫的嗅觉比人类灵敏14倍。猫的鼻腔内有2亿个嗅觉受体，数量甚至超过某些品种的狗。\n4. 味觉：猫在早期的演化过程中，因为基因突变，失去了甜的味觉，但它们不仅能感知酸，苦，咸味，选择适合自己口味的食物，还能尝出水的味道。\n5. 触觉：猫会在蹭人的时候释放特别的信息素。当这些特有的信息素留下时，其目的是在宣誓主权，提醒其他猫这是它们的区域。\n\n关于今天上海的天气，上海的天气预报显示，今天是阴天，露出了一些阳光，温度在11-15℃之间，当前有东北风，风力为3级，空气质量良好。'}

Process finished with exit code 0

"""