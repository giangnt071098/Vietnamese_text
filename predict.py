from keras.models import load_model
model = load_model("./model_0.0453_0.9854.h5")

def extract_phrases(text):
    pattern = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(pattern, text)

def guess(ngram):
    text = ' '.join(ngram)
    preds = model.predict(np.array([encode(text)]), verbose=0)
    return decode(preds[0], calc_argmax=True).strip('\x00')


def add_accent(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates)
    return output

def accent_sentence(sentence):
  list_phrases = extract_phrases(sentence)
  output = ""
  for phrases in list_phrases:
    if len(phrases.split()) < 2 or not re.match("\w[\w ]+", phrases):
      output += phrases
    else:
      output += add_accent(phrases)
      if phrases[-1] == " ":
        output += " "
  return output

text = '''Trung Quoc da mo rong anh huong cua ho trong khu vuc thong qua cac buoc leo thang ep buoc cac nuoc lang gieng o Hoa Dong, Bien Dong, boi dap dao nhan tao va quan su hoa cac cau truc dia ly tren Bien Dong trai luat phap quoc te; Tim cach chia re Hoa Ky khoi cac dong minh chau A thong qua cac no luc ep buoc va leo lai kinh te'''
print((accent_sentence(text)))