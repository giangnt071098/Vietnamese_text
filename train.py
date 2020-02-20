import re,os
import unidecode
import itertools
from nltk import ngrams
import string
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, LSTM, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
pass
with open("./train_data1.txt", "r",encoding='utf-8') as f_r:
    lines = f_r.read().split("\n")
    
print(len(lines))

alphabet = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'
MAXLEN = 30
NGRAM = 5
BATCH_SIZE = 1024

def remove_accent(text):
    return unidecode.unidecode(text)
def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)
phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]
def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)
    
list_ngrams = []
for p in tqdm(phrases):
  if not re.match(alphabet, p.lower()):
    continue
  for ngr in gen_ngrams(p, NGRAM):
    if len(" ".join(ngr)) < 32:
      list_ngrams.append(" ".join(ngr))
del phrases
list_ngrams = list(set(list_ngrams))


#list_ngrams = []

# for p in tqdm(phrases):
#   for ngr in gen_ngrams(p, NGRAM):
#     if len(" ".join(ngr)) < 30:
#       list_ngrams.append(" ".join(ngr))
# del phrases
# list_ngrams = list(set(list_ngrams))


accented_chars_vietnamese = [
    'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',
]
accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(accented_chars_vietnamese)))

def encode(text, maxlen=MAXLEN):
        text = "\x00" + text
        x = np.zeros((maxlen, len(alphabet)))
        for i, c in enumerate(text[:maxlen]):
            x[i, alphabet.index(c)] = 1
        if i < maxlen - 1:
          for j in range(i+1, maxlen):
            x[j, 0] = 1
        return x

def decode(x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)


HIDDEN_SIZE = 256

model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet)), return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()



train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=2019)
def generate_data(data, batch_size=128):
    cur_index = 0
    while True:
        
        x, y = [], []
        for i in range(batch_size):  
            y.append(encode(data[cur_index]))
            x.append(encode(unidecode.unidecode(data[cur_index])))
            cur_index += 1
            
            if cur_index > len(data)-1:
                cur_index = 0
        
        yield np.array(x), np.array(y)
train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

checkpointer = ModelCheckpoint(filepath=os.path.join('./model_{val_loss:.4f}_{val_acc:.4f}.h5'), save_best_only=True, verbose=1)
early = EarlyStopping(patience=2, verbose=1)
model.fit_generator(train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=10,
                    validation_data=validation_generator, validation_steps=len(valid_data)//BATCH_SIZE,
                    callbacks=[checkpointer, early])