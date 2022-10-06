#%%
#environnement bertoppic3
from inspect import stack
import re
from turtle import width
import streamlit as st
from PIL import Image 
import pandas as pd
import datetime as dt
import pickle
import plotly.express as px
from unidecode import unidecode

# import matplotlib.pyplot as plt
# from matplotlib.dates import DateFormatter
# import matplotlib.dates as mdates

# import BERTopic
import numpy as np
# cf https://github.com/conda-forge/bertopic-feedstock

import pyarrow
from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from umap import UMAP
# from sklearn.decomposition import PCA

#SECTION CONFIG  : LAYOUT & CONTEXT 
im = Image.open("./images/icon.png")
st.set_page_config(
  page_title="Topic detection",
  page_icon=im,
  layout="wide"
)
data_load_state = st.text('Loading data...')

#SECTION LOAD DATAFRAME
@st.cache(allow_output_mutation=True)
def load_data():
  data = pd.read_csv('df.csv', parse_dates=True).drop(columns = ['Unnamed: 0', 'month', 'year'])
  return data
df = load_data()
df = df[~df.test_message.isna()]
list_of_dates = df.date.to_list()
docs = df.new_message.to_list()

data_load_state.text("loading  nlp model. May take some time...")




# SECTION STYLE CODE BLOCKS & TERMINAL OUTPUT
from load_css import local_css
local_css("style.css")


terminal = "<div class='blue'>In your terminal : </div>" 
output = "<div class='red'>Output like : </div>"
_code_ = "<div class='green'>Python code : </div>"

#SECTION LOAD DATASET

st.title('Analysis and topic dectection of messages')

st.header('Process and milestones')
st.markdown("""---""")


st.header('Table of Contents', anchor = 'beginning')             
st.markdown('''
    ## 1. [Introduction ](#intro)
    ## 2. [NLP Preprocessing](#s2)
    ## 3. [Exploratory data analysis](#s3)
    ## 4. [Topic modeling](#s4)
    ## 5. [Enjoy our model prediction](#s5)
    ## 6. [Last experiment with guided modeling](#s6)
    ''', unsafe_allow_html=True)
st.markdown("""---""")

output = "<div class='red'>1. Introduction</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor='intro')
st.markdown("""
    **Topic modeling** is an **unsupervised Natural Language Processing (NLP) technique** used to identify recurring patterns of words from a collection of documents forming a text corpus.
    
    Our corpus consists of personal messages that come with the orders. We aim to detect the main topics in these messages
       
    Topics can be defined as “a repeating pattern of co-occurring terms in a corpus”.
    
    
    
    BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations 
    which obtains state-of-the-art results on a wide array of nlp tasks.
    
    In this app, we will introduce what we believe to be the most powerful topic modeling algorithm in the field today: BERTopic. *"BERTopic is a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions."* (Maarten Grootendorst)
    
    
    We will also include interactive visualizations to analyze huge amounts of information without needing to manually inspect any documents or topics at a granular level.
    
    ```
    Credentials : 
    
    @article{grootendorst2022bertopic,
    title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
    author={Grootendorst, Maarten},
    journal={arXiv preprint arXiv:2203.05794},
    year={2022}

    ```

    
    
  [back to top](#beginning)
  """)

output = "<div class='red'>2. NLP Preprocessing</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor= 's2')
st.markdown("""
            One of the most critical steps in any NLP-related project is column-wise preprocessing of all documents. 
            
            While there are some standard cleaning methods that, in general, can be applied to any corpus, there are also a considerable amount of intuition-driven decisions that we must make. 
            
            Each language requires a specific approach and these are some of the issues we had to deal with :
            
            * Does it make sense to **emojis**, do we need to convert them to text? In our case, the problem does not exist because there are very little of them (```array(['™', '®', '❤️', '❤', '©'], dtype=object)``` ) 
            
            
            * How to detect **named entities** (people, companies, places)? At what point should these named entities be deleted, without deleting "our" important words like "mum", or "granny"?Quick tip for french language : We chose to delete all equivalents of "tata Yoyo" and "mamie Nova", but we have kept tata and mamie :stuck_out_tongue_winking_eye:. As the messages have few words, we have ruled out deleting any of the > 100K first names in France, for performance reasons.
            
            * Should **numbers** be left as is, converted to text, or removed entirely? => removed
            
            * Should **words that occur in more than 80%** of all documents be ignored or investigated? => Investigated but kept except stopwords and words < 2 characters lenght
            
            * Should variations of the same root word be **lemmatized**? => Tested ! Worse results with stemming or lemmatization
            
            Given the very large size of the base dataset (> 2.5 million records) we use in this app a sample of data stratified by year, month, libevement
            
            At this stage of the analysis the main steps are as follows : 
            """)
## Run the below code if the check is checked ✅
if st.checkbox('Show me the code'):
  _code_ = "<div class='green'>Build a custom stopword list</div>"
  st.markdown(_code_, unsafe_allow_html=True)
  st.code('''
          
  # nltk corpus
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  nltk_corpus = set(stopwords.words("french"))

  # spacy corpus
  from spacy.lang.fr.stop_words import STOP_WORDS
  spacy_corpus = STOP_WORDS

  # french corpus 
  dic1 = {'une', 'un','cette', 'avec', 'qui','parce que', 'bonnes', 'bonne', 'bons', 'bon', 'fort', 'grosses','grosse','gros',  'petites','petit',  'petits', 'petit', 'pour', 'que', 'nous', 'notre', 'très','petit', 'gros', "a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait", "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autant", "autre", "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons", "b", "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr", "c", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra", "devrait", "different", "differentes", "differents", "différent", "différente", "différentes", "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc", "dont", "dos", "douze", "douzième", "dring", "droite", "du", "duquel", "durant", "dès", "début", "désormais", "e", "effet", "egale", "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers", "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes", "exactement", "excepté", "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient", "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc", "fois", "font", "force", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gens", "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i", "ici", "il", "ils", "importe", "j", "je", "jusqu", "jusque", "juste", "k", "l", "la", "laisser", "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale", "me", "meme", "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "mine", "minimale", "moi", "moi-meme", "moi-même", "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même", "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire", "necessairement", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment", "notre", "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh", "ohé", "ollé", "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce", "parfois", "parle", "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier", "particulière", "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne", "personnes", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce", "plein", "plouf", "plupart", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi", "pourrais", "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement", "pres", "probable", "probante", "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement", "rares", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste", "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sait", "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble", "semblent", "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si", "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient", "sois", "soit", "soixante", "sommes", "son", "sont", "sous", "souvent", "soyez", "soyons", "specifique", "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit", "suivant", "suivante", "suivantes", "suivants", "suivre", "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis", "tant", "tardive", "te", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "trois", "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes", "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur", "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà", "voire", "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut", "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"}
  print('french corpus : ', len(dic1))

  #build custom stop words list and dedup
  custom = [w for w in dic1| nltk_corpus | spacy_corpus  ]
  custom = list(set(custom))
  print('custom : ', len(custom))
          ''')
  st.markdown('''[back to top](#beginning)''')

  _code_ = "<div class='green'>Clean and lemmatize messages text for French</div>"
  st.markdown(_code_, unsafe_allow_html=True)
  st.code("""

  import string
  from unidecode import unidecode
  # Creating a dictionary with the keys being the name of the language and the values being the font
  # awesome icon.
  import re

  def clean(input_text):
    input_text = input_text.replace("'", ' ')
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))
    input_text = input_text.replace('!', '')              # to deal with any !

    input_text = re.sub(r'[0-9,.]+', '', input_text)      # Remove digits
    input_text = re.sub('\\n',' ', input_text)            # Remove line breaks
    input_text = re.sub('\s+', ' ', input_text).strip()   # Remove leading, trailing, and extra spaces
    input_text = re.sub("(.)\\1{3,}", "\\1", input_text)  # Remove duplicate caracters in string
    input_text = ' '.join([unidecode(word) for word in input_text.split(' ') if   (   (word.lower() not in custom) & (len(word) > 2)   ) ])
    return input_text
    
    # Real tests led us to remove the lemmatization step which contributed to generate  too many outliers in our corpus
    
    df['test_message'] = [clean(sentence) for sentence in df.message ]
    

  """)
  st.markdown('''[back to top](#beginning)''')

  _code_ = "<div class='green'>Detect Named Entities</div>"
  st.markdown(_code_, unsafe_allow_html=True)
  st.code(''' 
          # cf https://huggingface.co/BaptisteDoyen/camembert-base-xnli?candidateLabels=animal%2C+amiti%C3%A9%2C+science&multiClass=true&text=Scoubidou+est+un+bon+chien+que+j%27aime+beaucoup
  # Code available at https://colab.research.google.com/drive/1uEduA39f0wtMozJW7HhsfuRPnNIPS4jV#scrollTo=M-d75m30DF9g
  # Camembert-base model fine-tuned on french part of XNLI dataset.
  # One of the few Zero-Shot classification model working on french 🇫🇷

  from transformers import pipeline
  ner = pipeline("token-classification", model="Jean-Baptiste/camembert-ner")

  #Example of an output to be restructured
  %timeit ner("Bon anniversaire Tata Yoyo de la part de Caroline à la Réunion")
  16.8 ms ± 646 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  [{'entity': 'I-MISC',
    'score': 0.81861615,
    'index': 3,
    'word': '▁Tat',
    'start': 16,
    'end': 20},
  {'entity': 'I-MISC',
    'score': 0.80462503,
    'index': 4,
    'word': 'a',
    'start': 20,
    'end': 21},
  {'entity': 'I-MISC',
    'score': 0.8577907,
    'index': 5,
    'word': '▁Y',
    'start': 21,
    'end': 23},
  {'entity': 'I-MISC',
    'score': 0.8552661,
    'index': 6,
    'word': 'oyo',
    'start': 23,
    'end': 26},
  {'entity': 'I-PER',
    'score': 0.65235627,
    'index': 11,
    'word': '▁Caroline',
    'start': 40,
    'end': 49},
  {'entity': 'I-LOC',
    'score': 0.8488388,
    'index': 13,
    'word': '▁la',
    'start': 51,
    'end': 54},
  {'entity': 'I-LOC',
    'score': 0.99757963,
    'index': 14,
    'word': '▁Réunion',
    'start': 54,
    'end': 62}]

  # Detect any possible entity in unique messages with more than 2 words   
  name_entity_recognition = []
  for element in min_2_words_in_messages_to_analyse:
      entity = ner(element)
      name_entity_recognition.append([element, entity])

  #Aggregate all entities detected 
  #List  of unique types of entities : ['I-MISC', 'I-PER', 'I-LOC']

  import numpy as np
  import ast

  def get_people2(input_text, feature):
    """
    cleans text and extracts people from an input text
    """
    if len(input_text) > 0 :
      input_text = ast.literal_eval(input_text) #transforms back string to list of dict
      tokens_clean = []
      tokens = [] 
      last_pos = []
      end_pos = []
      for elem in input_text:
        if (elem['entity'] == feature) & (elem['score'] > 0.5):
          tokens.append(elem['word'])
          end_pos.append(elem['end'])
      k = 0
      for e, t in zip(end_pos, tokens ) : 
        last_pos.append(e)
        if k > 1:
          if last_pos[-1] - last_pos[-2] - len(t) != 0:
              t =  '|' + t
        k = k+1
        tokens_clean.append(t)
      result =  ''.join(tokens_clean)
      result = result.split('|')
      result = [e.strip('▁') for e in result]
      return ', '.join(result)
    else:
      return ""
  # Create a new column with the person first name and or last name entities
  df['people'] = [get_people2(x, 'I-PER') for x in df.ner_results]
    
  ### Manually create a list of all words in people named entities related to 'family'
  family = [ 'peres', 'mman','mamoune',  'lapin', 'darling','babe', 'femmes','papy', 'nany', 'filles', 'mam', 'sister', 
          'bellemaman', 'maminette', 'mama','famille','moman', 'fils','marraine', 'doudou', 'puce', 'kisses', 'bebe',
          'nana',  'mother', 'mummy','granny','enfants', 'mere','soeur', 'mamy','fille','femme', 'papa', 'mami', 'coco', 
          'grandmeres',  'meme', 'mamies', 'tata',  'dad', 'valentines',  'princesse',
          'valentin','love', 'amour', 'grandmeres', 'cherie','mamounette',  'bonne','papa',  'maman cherie',  'soeurette',  
          'amour',  'mamina',  'ma maman',  'maminette', 
          'maman cherie damour',  'mman',  'pepe',  'bisous', 'bisou',
          'mamy',  'maminou',  'ma soeur',  'cousine',  'monsieur',  'mamie damour',  'mere',  'mamour',  
          'mamita',  'maman',  'docteur',  'belle maman',  'ange',  'maminoumame',  'mamam',  'mme',  'papa bisous',  
          'mamie maman',  'bellemere',  'memere',  'fete mamie',  'petite maman',  'ton cheri',  'maman mamie',  
          'moumou',  'mamanmamie',  'mame',  'mamou',  'mamie d',  'e meme',  'poussin',  'fils',  'mamilou',  
          'grandsmeres',  'tonton',  'bellemam',  'marraine',  'coco',  'mari',  'mamounette',  'mum',  'mamie bisous', 
          'frerot',  'mamie',  'papy',  'mounette',  'mariee',  'mamie',  'presidente',  'bellemaman',  'mamounette damour', 
          'mamoun',  'papa maman',  'tati',  'mamie',  'mamoune',  'bellesoeur',  'mamie gros',  'maman papa',  
          'mamanie',  'maman maman',  'tatie',  'maman d',  'm',  'ma cherie',  'ami',  'frere',  'tante',  'mamy gros',  'mama', 
          'mamie cherie',  'maman damour',  'grand mere',  'mamie mamie',  'papounet',  'tata damour',  'mamans',  'mamie papy',  
          'puce',  'maman m',  'maman je taime',  'directeur',  'fille',  'bonne maman',  'mamounette cherie',  'bon',
          'moman',  'tata',  'grandmaman',  'bisou',  'cherie',  'ma chere maman',  'meme',  'mr',  'mamie papi',  'daddy',
          'gros',  'moumoune',  'bibiche',  'mon amour',  'grandmere',  'soeur',  'grand mamie',  'niece',  'marie', 
          'dr',  'coeur',  'mamee',  'tantine',  'parrain',  'mam',  'princesse',  'mams',  'dame',  'mami',  'ton fils',  'madame',  'papi',  'namour']
  
  ### Clean  the people column with no family word
  df['people_cleaned'] = [ast.literal_eval(x)  for x in df.people]
  df['people_cleaned'] = [', '.join(x) if x != "['']" else '' for x in df.people_cleaned]
  df['people_cleaned'] = [unidecode(x.lower()) for x in df.people_cleaned]
  def remove_word(sentence):
    cleaned_sentence = []
    sentence = sentence.split(', ')
    for word in sentence:
      if word not in family:
        cleaned_sentence.append(word)
    return ', '.join(cleaned_sentence)
  df['people_cleaned'] = [remove_word(sentence) for sentence in df.people_cleaned if sentence is not None]  
  
  #### Clean the text messages :
  def remove_people(input_text, people) : 
  if str(people) != 'nan':
    input_text = input_text.split(' ')
    people = people.split(', ')
    return " ".join([x for x in input_text if unidecode(x.lower()) not in people])
  else:
    return input_text


df['test_message'] = [remove_people(x, y) for x, y in zip(df.test_message, df.people_cleaned)]
df['test_message'] = [ x.lower() for x in df.test_message] #easier for clustering
df = df[(df.test_message != '') & (~df.test_message.isna())] #no blank records ;-)

          ''')
  st.markdown('''[back to top](#beginning)''')


st.markdown("""---""")

#SECTION EDA
output = "<div class='red'>3. Exploratory data analysis</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor='s3')
st.markdown('**Frequency of messages over time**')
fig = px.line(df.groupby(by='date')['message'].size(), 
              x=df.groupby(by='date')['message'].size().index, 
              y=df.groupby(by='date')['message'].size().values, 
              title='Frequency  of messages over time')
st.plotly_chart(fig, use_container_width=True)

## Run the below code if the check is checked ✅
if st.checkbox('Show raw demo data'):
    st.subheader('Raw data')
    st.write(df.sample(10))    
## export du dataframe

    export =df.to_csv()

    st.download_button(label='📥 Download raw data',
                                    data=export ,
                                    file_name= 'demo_data.csv')


st.markdown('**Message length distribution**')
df['message_length'] = [len(x.split()) for x in df.message]
longest = 'The longest message has ' + str(df.message_length.max()) + ' words'
st.write(longest)
hist_data = [df['message_length'].values]
group_labels = ['distplot of messages lengths'] # name of the dataset
import plotly.figure_factory as ff
fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1,0.2, 0.3,  0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1])
st.plotly_chart(fig, use_container_width=True, )
st.markdown(''''

  **Conclusion:**
  It's not much to get the context, but it's good for using zero-shot deep-learning models !
            ''')

st.markdown('''
  **Content analysis with WordClouds:**
  
  WordClouds are useful tools for summarising the most important concepts in a text, a web page or a book. The more a word is present in the text under consideration, the larger it appears in the wordcloud.
  Please see below a wordcloud for all messages
            ''')

df['lib_event'] = df.libevenement
df['lib_event'] = df.lib_event.map({
  "Anniversaire": "anniversaire" ,
  "Pour le plaisir": "pour_le_plaisir" ,
  "Amour": "amour" ,
  "Remerciements": "remerciements" ,
  "Deuil": "deuil" ,
  "Naissance": "naissance" ,
  "1er Mai": "1er_mai" ,
  "Bon rétablissement": "bon_retablissement" ,
  "Mariage": "mariage" ,
  "Félicitation": "felicitation" ,
  "Toussaint": "toussaint" ,
  "Fêtes de fin d'année": "fetes_de_fin_d_annee" ,
  "Fête du prénom": "fete_du_prenom" ,
  "Fête des grand-mères": "fete_des_grand-meres" ,
  "Fête des mères": "fete_des_meres" ,
  "Anniversaire de Mariage": "anniversaire_de_mariage" ,
  "Saint-Valentin": "saint-valentin" ,
  "Pâques": "paques" ,
  "Fête des pères": "fete_des_peres" ,
  "Départ en retraite": "depart_en_retraite" })

options = sorted(df.libevenement.unique())[1:]

current_image = st.selectbox('select your event to see the corresponding wordcloud', 
                  options = options)
current_image = './wordclouds/' + unidecode(current_image).replace(' ', '_').replace("'", '_').lower() + '.png'


st.image(current_image)

st.markdown('''
          **Conclusion** : 
          
          Although very practical, WordClouds are not always the most relevant and therefore the most effective tool for conducting textual analyses. 
          They are less precise than a bar chart, which gives more specific indications on the frequency of words and allows an effective comparison of the frequency of appearance of words in the text.  They therefore have little precision in the information they convey. 
          Also, it will not be easy to translate the context in which these words appear (and this is why we have embedded the image in a specific form to show the dimension of the context).
            ''')

st.markdown('[back to top](#beginning)')
output = "<div class='red'>4. Topic modeling</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor='s4')
st.subheader('''4.1 The mechanism ''')
st.markdown('''

Before we dive into the modeling process, let us have a look at the mechanism. 

State-of-the-art topic modelling is a clustering task using pre-trained transformer-based 🤗 language models to embed each document

**Embeddings** can then be compared e.g. with cosine-similarity to find sentences with a similar meaning.

More specifically, we generate document embedding with pre-trained transformer-based language models, clusters these embeddings using HDBSCAN to generate the clusters, and finally, generates topic representations with a class-based TF-IDF procedure.

In their nature, the messages are very similar to the experiment conducted at Lincoln, whose main objective was to implement NLP from scratch techniques on a corpus of messages from a Twitch chat to get the topic.
The messages are expressed in French, but on an internet platform with the internet vocabulary that this implies (slang, mistakes, community vocabulary, private jokes ...). 
Overall, the approach gives satisfactory results in identifying recurring "similar" messages. We applied Lincoln's twich-sentence-transformer model to our corpus to get better results. The bonus of this approach is to have consistent clusters, with less than 20% of outliers (against > 50% for other linguistic models).

**The steps look like this:**

* First, we load specific embeddings for our corpus through  Lincoln twich sentence transformers..
* Then, the BERT model generates a representation vector for each document.
* Next, the HDBSCAN algorithm is used for the clustering process. Therefore, each group contains texts that have a similar meaning to them.
* After that, the c-TF-IDF algorithm retrieves the most relevant words for each topic.
* Finally, to maximize the diversity, the Maximize Candidate Relevance algorithm is used.
            


       
            ''')
st.image('./images/bert_modeling.png')
st.markdown('*Source: The BERTopic Documentation*')

st.markdown('''
In order to get an accurate representation of the topics from our bag-of-words matrix, 
TF-IDF is adjusted to work on a cluster/categorical/topic-level instead of a document-level. 
This adjusted TF-IDF representation is called c-TF-IDF takes into account what makes the documents in once cluster different from documents in another cluster.   
            
            
If you are unfamiliar with TF-IDF in the first place, all you need to know in order to generally grasp what is going on here is one thing: 
it allows for comparing the importance of words between documents by computing the frequency of a word in a given document 
and also the measure of how prevalent the word is in the entire corpus. 



Now, if we instead treat all documents in a single cluster as a single document and then perform TF-IDF, 
the result would be importance scores for words within a cluster. 
The more important words are within a cluster, the more representative they are of that topic. 
            ''')
st.image('./images/c-TF-IDF.svg')
st.markdown('''
**c-TF-IDF formula:**

Each cluster is converted to a single document instead of a set of documents. 
We extract the frequency of word x in class c, where c refers to the cluster we created before. 
This representation is L1-normalized to account for the differences in topic sizes.

Then, we take take the logarithm of one plus the average number of words per class A divided by the frequency of word x across all classes. 
We add plus one within the logarithm to force values to be positive. 
This results in our class-based idf representation. We then multiply tf with idf to get the importance score per word in each class. 

''')







#SECTION TOPIC MODELING


@st.cache(allow_output_mutation = True)
def load_model():
  model= pickle.load(open('./pickled/model.pkl', 'rb'))
  return model

def load_topics():
  topics = pickle.load(open('./pickled/topics.pkl', 'rb'))
  return topics

def load_embeddings():
  embeddings = pickle.load(open('./pickled/embeddings.pkl', 'rb'))
  return embeddings

def load_topics_over_time():
  topics_over_time = pickle.load(open('./pickled/topics_over_time.pkl', 'rb'))
  return topics_over_time

model = load_model()        
topics = load_topics()
embeddings = load_embeddings()
topics_over_time = load_topics_over_time()


# Run the below code if the check is checked ✅
if st.checkbox('Check this box to get the code... or not!'):
  _code_ = "<div class='green'>Bertopic modeling </div>"
  st.markdown(_code_, unsafe_allow_html=True)
  st.code('''         
#Prepare our 3 variables :     
list_of_dates = df.date.to_list()
docs = df.test_message.to_list()
classes = df.libevenement.values

#Instantiate our sentence transformer model with pre-trained "Lincoln twich" model
sentence_model = SentenceTransformer("lincoln/2021twitchfr-conv-bert-small-mlm-simcse")

# After having calculated our top n words per topic there might be many words 
# that essentially mean the same thing. 
# As a little bonus, we can use the diversity parameter in BERTopic 
# to diversity words in each topic such that we limit the number of duplicate words 
# we find in each topic. This is done using an algorithm called Maximal Marginal Relevance 
# which compares word embeddings with the topic embedding.
# We do this by specifying a value between 0 and 1, with 0 being not at all diverse 
# and 1 being completely diverse:



from hdbscan import HDBSCAN
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True, min_samples=5)
model = BERTopic(
    language = 'french',
    hdbscan_model=hdbscan_model,
    embedding_model=sentence_model, 
                        min_topic_size=50, 
                        # n_gram_range=(1,2), 
                        diversity=0,
                        
                 )
embeddings = sentence_model.encode(docs, show_progress_bar=False)

#Takes ~ 20 minutes to be completed
topics_over_time = model.topics_over_time(docs, timestamps=list_of_dates,
                                          nr_bins=18, 
                                          evolution_tuning = True, 
                                          global_tuning=True
                                          )


topics, probs = model.fit_transform(docs, embeddings)

#Save our models, topics, embeddings for further use
pickle.dump(model, open('./pickle/model.pkl', 'wb'))
pickle.dump(topics, open('./pickle/topics.pkl', 'wb'))
pickle.dump(embeddings, open('./pickle/embeddings.pkl', 'wb'))
pickle.dump(topics_over_time, open('./pickle/topics_over_time.pkl', 'wb'))
         
         ''')
st.subheader('4.2 Topics exploration')
  
st.markdown(''' Now we can retrieve topics among all messages''')


st.write('''
         The table below has 3 main columns, providing information about all the 75 topics in descending order of topics size/Count.

**Topic** is the topic number, a kind of identifier. Topic -1 denotes the topic consisting of outlier documents which are typically ignored due to terms having a relatively high prevalence across the whole corpus and thus, low specificity toward any cohesive theme or topic.

**Count** is the number of words in the topic.

**Name** is the name given to the topic. For each topic, we can retrieve the top 4 words and their corresponding c-TF-IDF score. The higher the score, the most relevant the word is in representing the topic.
         '''
         )


st.write(model.get_topic_info())

topic_number = model.get_topic_info().Topic.values[1:]


current_topic = st.selectbox('Select a topic to get its 10 most significant words and score 👇', 
                  options = topic_number)


st.write(model.get_topic(current_topic))

st.subheader('4.3 Quick top 12 topics benchmark (> 50% of messages count)')
st.markdown('''           
Topic visualization helps in gaining more insight about each topic.

The most relevant words of each topic can be visualized in a form of barchart out of the c-TF-IDF score, which is interesting to visually compare topics.
Below is the corresponding visualization for the top 16 topics.''')

fig = model.visualize_barchart(top_n_topics=12)
st.plotly_chart(fig)
st.markdown('---')
st.subheader('4.4 Documents and topics')
st.markdown(''' 
            
            Let's visualize the topics and get insight into their relationships.To do so, we can use the topic_model.visualize_documents() function. 
            This function recalculates the document embeddings and reduces them to 2-dimensional space for easier visualization purposes.
            
            However, you might want a more fine-grained approach where we can visualize the documents inside the topics to see if they were assigned correctly or whether they make sense. 
            
            
            **To zoom in**  : click on the map than draw a rectangle on the map to get into details 
             
             **To zoom out**  double click anywhere on the map. 
             
             You can also double click on items in the legend to disable their viewing in the map
             
             ''')
fig_map = model.visualize_documents(
                                docs, 
                                embeddings = embeddings, 
                                topics = topics, #[i for i in  range(40)],
                                hide_document_hover = False,
                                hide_annotations = False,height=1200)
st.plotly_chart(fig_map, use_container_width=True )


st.subheader('4.5 Intertopic Distance map')
st.markdown('''
          The intertopic distance map  below is a visualization of the topics in a two-dimensional space. 
          
          The area of these topic circles is proportional to the amount of documents that belong to each topic. 
          
         Topics that are closer together have more words in common.
         
         You can zoom in the map and zoom out with a double click.
         
         You can use the slider to select the topic which then lights up red. If you hover over a topic, then general information is given about the topic, including the size of the topic and its corresponding words.''')
fig_map = model.visualize_topics(height = 600)
st.plotly_chart(fig_map, use_container_width=True)



st.subheader('4.6 Hierarchical clustering')
st.markdown('''
Hierarchical clustering, also known as hierarchical cluster analysis, is an algorithm that groups similar objects into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other. 

Hierarchical clustering starts by treating each observation as a separate cluster. Then, it repeatedly executes the following two steps: (1) identify the two clusters that are closest together, and (2) merge the two most similar clusters. This iterative process continues until all the clusters are merged together. This is illustrated in the diagrams below.
           
            ''')
st.image('./images/Hierarchical-clustering-3-1.webp')

from scipy.cluster import hierarchy as sch

# Hierarchical topics
linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
hierarchical_topics = model.hierarchical_topics(docs, linkage_function=linkage_function)
fig_hierarchical_clustering = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
st.plotly_chart(fig_hierarchical_clustering, use_container_width=True)


# similar_topics, similarity = model.find_topics("papa", top_n = 10)
# print(similar_topics)

st.markdown(''' 
After seeing the potential hierarchy of our topics, this allows us to merge specific topic such as
```
anniversaire = [0, 2, 16, 36, 27,20,53]
deuil = [26, 15, 23, 38, 24]
fete = [1,  6,  4, 28 ]
felicitation_naissance = [47,17]
valentin = [8, 64]
amour  = [16, 2, 0, 27, 38, 18]
```
''')

            
# topics_to_merge = [[0, 2, 16, 36, 27,20,53],
# [26, 15, 23, 38, 24],
# [1,  6,  4, 28 ],
# [47,17],
# [8, 64],
# [16, 2, 0, 27, 38, 18]]
# model.merge_topics(docs, topics_to_merge)

# st.write('please see below the updated map')

# fig_map_reduced = model.visualize_documents(docs, 
#                                  embeddings = embeddings, 
#                                  topics = topics,
#                           hide_document_hover = False,
#                           hide_annotations = False)
# st.plotly_chart(fig_map_reduced)




# st.write('Topics over time')
# fig_over_time = model.visualize_topics_over_time(topics_over_time, topics = [0,1,2])
# st.plotly_chart(fig_over_time)


st.write('''
One issue with the approach above is that it will merge topics regardless of whether they are very similar. 
They are simply the most similar out of all options. This can be resolved by reducing the number of topics automatically. 

To do this, we can use HDBSCAN to cluster our topics using each c-TF-IDF representation. 
Then, we merge topics that are clustered together. 
Another benefit of HDBSCAN is that it generates outliers. These outliers prevent topics from being merged if no other topics are similar.     
         ''')




# ########################Test pour tenter de régler le pb




st.subheader('4.7 Document map after cluster reduction')

# # 
# model2 = model.reduce_topics(docs, nr_topics=20) 
# pickle.dump(model2, open('first_reduced_model.pkl', 'wb'))
model2 = pickle.load(open('./pickled/first_reduced_model.pkl', 'rb'))

st.write(model2.get_topic_info())
fig_map_reduced2 = model2.visualize_documents(docs, 
                                 embeddings = embeddings, 
                                 topics = topics,
                          hide_document_hover = False,
                          hide_annotations = False, height= 1200)
st.plotly_chart(fig_map_reduced2, use_container_width=True)


st.subheader('4.8 Topics over time after  cluster reduction')
st.markdown('''
You can select / deselect items in the legend to get the desired benchmark
            ''')


fig_test = model2.visualize_topics_over_time(topics_over_time, topics = [i for i in range(20)], height=600)
st.plotly_chart(fig_test)


#intertopic distance map
fig_map_reduced = model2.visualize_topics(height = 800)
st.plotly_chart(fig_map_reduced, use_container_width=True)


# Hierarchical topics
linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
hierarchical_topics_reduced = model2.hierarchical_topics(docs, linkage_function=linkage_function)
fig_hierarchical_clustering_reduced = model2.visualize_hierarchy(hierarchical_topics=hierarchical_topics_reduced)
st.plotly_chart(fig_hierarchical_clustering_reduced, use_container_width=True)

st.subheader('4.9 Topic representation across our categories')

st.markdown(''' 
            We are of course interested in how certain topics are represented over our own categories (fête des grands-mères, Saint Valentin, ...).
            
            Now that we have created our global topic model, let us calculate the topic representations across each category:''')

topics_per_class = model2.topics_per_class(docs, classes=df.libevenement.values)
fig3 = model2.visualize_topics_per_class(topics_per_class, top_n_topics=20)
st.plotly_chart(fig3)

st.markdown('''
            You can hover over the bars to see the topic representation per class.''')



st.markdown('[back to top](#beginning)')


output = "<div class='red'>5. Enjoy our model predict! (with topic reduction)</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor='s5')
new_docs = st.text_input('Type below a sample message', 'Joyeux anniversaire maman !!!')


pred, probs = model2.transform(new_docs)
topic_words = model2.get_topic(pred[0])

result = 'Predicted topic : '  + str(pred[0])  + ' with a probability of ' +  str(probs[0])
# st.write(result)
# st.write(pred[0])
# st.write(probs)
# st.write(pd.DataFrame(topic_words))


data = model2.get_topic_info()
if data[data.Topic == pred[0]].Topic.values[0] == -1:
      st.write('Sorry this is an outlier document')
else :  
  st.write(result)
  st.write('Below is the most similar topic :')
  st.write(data[data.Topic == pred[0]])



st.markdown('''[back to top](#beginning)''')
st.markdown("""---""")


#<------------------- A debuguer ---------->
# #SECTION LAST MODEL
output = "<div class='red'>6. Last experiment with guided modeling</div>"
st.markdown(output, unsafe_allow_html=True)
st.subheader(""" """, anchor='s6')

st.markdown('''
In the process of modelling we have tested different sentence transformers models in order to minimise outliers, including
*paraphrase-MiniLM-L12-v2* or  *dangvantuan/sentence-camembert-large*. 

We have performed tests with and without data cleaning to get the best one.

At this stage, most relevant results are obtained with the sentence transformer *embedding lincoln/2021twitchfr-conv-bert-small-mlm-simcse* and a deletion of named entities and stop words.

We propose a third model for analysis, without prior any data cleaning, but with a guided modeling approach.

**Guided Topic Modeling** or **Seeded Topic Modeling** is a collection of techniques that guides the topic modeling approach by setting a number of seed topics in which the model will converge to. These techniques allow us to set a pre-defined number of topic representations that are sure to be in documents.

First, we create embeddings for each seeded topics by joining them and passing them through the document embedder. These embeddings will be compared with the existing document embeddings through cosine similarity and assigned a label. If the document is most similar to a seeded topic, then it will get that topic's label. If it is most similar to the average document embedding, it will get the -1 label. These labels are then passed through UMAP to create a semi-supervised approach that should nudge the topic creation to the seeded topics.

Second, we take all words in seed_topic_list and assign them a multiplier larger than 1. Those multipliers will be used to increase the IDF values of the words across all topics thereby increasing the likelihood that a seeded topic word will appear in a topic. This does, however, also increase the chance of an irrelevant topic having unrelated words. In practice, this should not be an issue since the IDF value is likely to remain low regardless of the multiplier. The multiplier is now a fixed value but may change to something more elegant, like taking the distribution of IDF values and its position into account when defining the multiplier.

We also fine-tune the topic extraction with BM25 weighting cf https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html and reduce frequent words.

We also performed this approach on a pre-cleaned corpus but results are more consistant on raw messages.

Here is the list for the guided modeling
```
seed = [
        ['anniversaire', 'birthday'],
        ['saint-valentin', 'saint valentin', 'valentin'],
        ['joyeux Noël', 'noel'],
        ['bonne année'],
        ['paques'], 
        ['condoléances', 'deuil', 'tristesse', 'peine', 'triste','décès', 'prières', 'épreuve', 'hommage', 'regrets', 'prions' ],
        ['mariage'],
        ['merci', 'remerciements']
                ]
```


)
''')
## Run the below code if the check is checked ✅
if st.checkbox('Code for Guided Topic Modeling'):
  _code_ = "<div class='green'>Custom TFIDF, fine-tuned with BM25 weighting and seeded topic modeling, </div>"
  st.markdown(_code_, unsafe_allow_html=True)
  st.code('''
df = df[(df.message != '') & (~df.message.isna())] #no empty messages !
list_of_dates = df.date.to_list()
docs = [str(elem) for elem in df.message]
list_of_dates = df.date.to_list()
docs = df.test_message.to_list()

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

# cf https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html#visual-overview
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

sentence_model = SentenceTransformer("lincoln/2021twitchfr-conv-bert-small-mlm-simcse")

from hdbscan import HDBSCAN



hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True, min_samples=5)
seed = [['anniversaire', 'birthday'],
    ['saint-valentin', 'saint valentin', 'valentin'],
    ['joyeux Noël', 'noel'],
    ['bonne année'],
    ['paques'], 
    ['condoléances', 'deuil', 'tristesse', 'peine', 'triste','décès', 'prières', 'épreuve', 'hommage', 'regrets', 'prions' ],
    ['mariage'],
    ['merci', 'remerciements']
                ]

model = BERTopic(
    language = 'french',
    ctfidf_model=ctfidf_model,
    hdbscan_model=hdbscan_model,
    embedding_model=sentence_model, 
    min_topic_size=50, 
    # n_gram_range=(1,2), 
    diversity=0,
    # umap_model=umap_model,
    seed_topic_list=seed
                 )
embeddings = sentence_model.encode(docs, show_progress_bar=False)

topics, probs = model.fit_transform(docs, embeddings)''')
  
@st.cache(allow_output_mutation = True)
def load_model_best():
  model= pickle.load(open('./pickled/last/model.pkl', 'rb'))
  return model

def load_topics_best():
  topics = pickle.load(open('./pickled/last/topics.pkl', 'rb'))
  return topics

def load_embeddings_best():
  embeddings = pickle.load(open('./pickled/last/embeddings.pkl', 'rb'))
  return embeddings

def load_topics_over_time_best():
  topics_over_time = pickle.load(open('./pickled/last/topics_over_time.pkl', 'rb'))
  return topics_over_time

def load_reduced_model_best():
  reduced_model = pickle.load(open('./pickled/last/model_reduced.pkl', 'rb'))
  return reduced_model

def load_reduced_over_time_best():
  topics_over_time_reduced = pickle.load(open('./pickled/last/topics_over_time_reduced.pkl', 'rb'))
  return topics_over_time_reduced

model_best              = load_model_best()        
topics_best             = load_topics_best()
embeddings_best         = load_embeddings_best()
topics_over_time_best   = load_topics_over_time_best()
reduced_model           = load_reduced_model_best()
topics_over_time_reduced = load_reduced_over_time_best()


docs_message            = [str(elem) for elem in df.message]

st.subheader('6.1 Topics frequency with Guided Topic Modeling ')
st.write(model_best.get_topic_freq().head(40))
st.write('At this stage, outliers account for less than 20% of messages')


st.subheader('6.2 Documents & Topics with Guided Topic Modeling  (top 40)')
map_fig_guided = model_best.visualize_documents(docs_message, 
                          embeddings = embeddings, 
                          topics = [i for i in range(40)], 
                          hide_document_hover = False,
                          hide_annotations = False,
                          height = 1200)
st.plotly_chart(map_fig_guided,  use_container_width=True)

st.subheader('6.3 Intertopic Distance Map')
fig_guided_distance = model_best.visualize_topics()
st.plotly_chart(fig_guided_distance,  use_container_width=True)

st.subheader('6.4 Topic word scores')
fig_bar_chart = model_best.visualize_barchart(top_n_topics=12, n_words=10, height=300)
st.plotly_chart(fig_bar_chart,  use_container_width=True)

st.subheader('6.5 Main topics over time')
guided_fig_over_time = model_best.visualize_topics_over_time(topics_over_time)
st.plotly_chart(guided_fig_over_time,  use_container_width=True)

st.subheader('6.6 Documents map after topic reduction')

st.write(reduced_model.get_topic_freq())
map_reduced = reduced_model.visualize_documents(
  docs_message, 
  embeddings = embeddings, 
  topics = topics,
  hide_document_hover = False,
  hide_annotations = False, 
  height = 1200)
st.plotly_chart(map_reduced,  use_container_width=True)

fig_intertopic_reduced_map = reduced_model.visualize_topics()
st.plotly_chart(fig_intertopic_reduced_map,  use_container_width=True)

st.subheader('6.7 Topics per class after topic reduction')

topics_per_class = reduced_model.topics_per_class(docs_message, classes=df.libevenement.values)
fig_topics_per_class = model.visualize_topics_per_class(topics_per_class)
st.plotly_chart(fig_topics_per_class, use_container_width=True)



st.markdown("""---""")
st.markdown(''' 
            Models have been trained on a Google Colab Pro session with 37.8 GB of Ram. Training a model takes about 7 /30 minutes per model versus > 6-8 hours on a laptop.
            
            Trained models are pickled and "reused" in this app for greater performance.
            ''')

# %%
