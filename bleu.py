from nltk.translate import bleu_score
import sentencepiece as spm
import custom_tokenizer
from pathlib import Path
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction

extra_vocab = ['<unk>', '<s>', '</s>', '<mask>', '<pad>']
CustomTextEncoder = custom_tokenizer.CustomTextEncoder
vocab_prefix = './dataset/m'
chencherry = SmoothingFunction()

def gen_tokenizer(prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + '.model')
    encoder = CustomTextEncoder(sp)
    return encoder

def  calc_bleu(encoder, orig, gen):
    hy = encoder.encode_words(orig)
    re = encoder.encode_words(gen)
    res = [re]
    bleu = bleu_score.sentence_bleu(res, hy,
                                    smoothing_function=chencherry.method1)
    return bleu

def read_bleus(encoder, path):
    df = pd.read_csv(path, header=None)
    count = 0
    bleu_sum = 0
    for sents in df.itertuples():
        count += 1
        bleu_sum += calc_bleu(encoder, sents[1], sents[2])
    bleu_mean = bleu_sum / count
    return bleu_mean

def read_bleus_one(encoder, path):
    df = pd.read_csv(path, header=None)
    bleu_max = 0
    count = 0
    for sents in df.itertuples():
        if calc_bleu(encoder, sents[1], sents[2]) == 1.0:
            bleu_max += 1
        count += 1
    return bleu_max / count

def read_bleus_sample(encoder, path):
    df = pd.read_csv(path, header=None)
    max_sample = []
    elif_sample = []
    else_sample = []
    count = 0
    for sents in df.itertuples():
        if sents[1] == sents[2]: # calc_bleu(encoder, sents[1], sents[2]) == 1.0:
            max_sample.append([' '.join(sents[1]), ' '.join(sents[2])])
        elif calc_bleu(encoder, sents[1], sents[2]) < 1e-9:
            elif_sample.append([' '.join(sents[1]), ' '.join(sents[2])])
        else:
            else_sample.append([' '.join(sents[1]), ' '.join(sents[2])])
        count += 1
    return max_sample, elif_sample, else_sample



def main():
    encoder = gen_tokenizer(vocab_prefix)
    path = './dataset/lm_ae_performance.csv'
    bleu_mean = read_bleus(encoder, path)
    print(bleu_mean)
    # 0.5032865387228846
    bleu_max = read_bleus_one(encoder, path)
    print(bleu_max)
    # 0.3185
    ms, elifs, elses = read_bleus_sample(encoder, path)

    import csv
    with open('max_sample.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(ms)
    with open('elif_sample.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(elifs)
    with open('else_sample.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(elses)

if __name__ == '__main__':
    main()
