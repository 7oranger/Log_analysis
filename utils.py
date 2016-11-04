def load_dic(dic_file):
    idx2word = []
    word2idx = {}
    i = 0
    with open(dic_file, "rb") as f:
        for line in f:
            word = line.strip().split(",")[0]
            idx2word.append(word)
            word2idx[word] = i
            i += 1

    return idx2word, word2idx


def sent2vector(sentence, word2idx):
    retvector = []
    for word in sentence.strip().split(" "):
        retvector.append(word2idx[word])
    return retvector


def vector2sent(vector, idx2word):
    sentence = [idx2word[word] for word in vector]
    return " ".join(sentence)


if __name__ == "__main__":
    idx2word, word2idx = load_dic("data/dic.txt")
    f = open("data/all.txt", "rb")
    context = f.readlines()[0:10]
    for line in context:
        print vector2sent(sent2vector(line, word2idx),idx2word)
