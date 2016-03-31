from load_data import load_CVAT_2
filename = './resources/CVAT (utf-8).csv'
texts, valence, arousal = load_CVAT_2(filename, categorical="all")
len_text = []
from CKIP_tokenizer import segsentence
out = []
for idx, i in enumerate(texts):
    # print(list(i))
    print(idx)
    out.append(" ".join(segsentence(i)))
    # len_text.append(len(.split()))
from save_data import dump_picle
dump_picle(out, "tokenized_texts_(newest3.31).p")
print("The tokenized text is saved.")