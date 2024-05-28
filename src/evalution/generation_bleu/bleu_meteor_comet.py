import argparse
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import json
from comet.models import download_model, load_from_checkpoint

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", '--sbn_file', default="/Users/xiaozhang/code/Parallel-Meaning-Bank-5/data/pmb-5.1.0/seq2seq/it/test/standard.sbn", type=str,
                        help="file path of first sbn, one independent sbn should be in one line")
    parser.add_argument("-s2", '--sbn_file2', default="/Users/xiaozhang/code/Parallel-Meaning-Bank-5/results/generation_pre_train/it/facebook-mbart-large-50/standard.sbn", type=str,
                        help="file path of second sbn, one independent sbn should be in one line")
    args = parser.parse_args()
    return args


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        l = []
        for line in file.readlines():
            l.append(line.split('\t')[0].strip())
        return l


def calculate_sentence_bleu(reference_file, candidate_file):
    # Read and tokenize the reference and candidate sentences
    references = [word_tokenize(sentence.strip().lower()) for sentence in read_file(reference_file)]
    candidates = [word_tokenize(sentence.strip().lower()) for sentence in read_file(candidate_file)]

    # Ensure both files have the same number of sentences
    if len(references) != len(candidates):
        raise ValueError("The number of sentences in both files must be the same")

    # Calculate and print BLEU scores for each sentence
    scores = []
    for ref, cand in zip(references, candidates):
        score = sentence_bleu([ref], cand, weights=[1,0,0,0])  # reference needs to be in a list of lists
        scores.append(score)
    return scores


def calculate_sentence_meteor(reference_file, candidate_file):
    references = [word_tokenize(sentence.strip().lower()) for sentence in read_file(reference_file)]
    candidates = [word_tokenize(sentence.strip().lower()) for sentence in read_file(candidate_file)]

    if len(references) != len(candidates):
        raise ValueError("The number of sentences in both files must be the same")

    scores = []
    for ref, cand in zip(references, candidates):
        score = meteor_score([ref], cand)
        scores.append(score)
    return scores


def calculate_sentence_comet(reference_file, candidate_file):
    model = load_from_checkpoint(download_model("wmt20-comet-da"))

    references = [line.strip() for line in read_file(reference_file)]
    candidates = [line.strip() for line in read_file(candidate_file)]

    if len(references) != len(candidates):
        raise ValueError("The number of sentences in both files must be the same")

    # 准备数据
    data = [
        {"src": "The source text which is fixed or ignored for monolingual evaluation.",
         "mt": cand,
         "ref": ref}
        for ref, cand in zip(references, candidates)
    ]

    # 计算COMET分数
    scores = model.predict(data, batch_size=8, gpus=0)
    return scores["scores"]


if __name__ == '__main__':
    # Example usage
    args = create_arg_parser()
    bleu_scores = calculate_sentence_bleu(args.sbn_file, args.sbn_file2)
    meteor_scores = calculate_sentence_meteor(args.sbn_file, args.sbn_file2)
    comet_scores = calculate_sentence_comet(args.sbn_file, args.sbn_file2)

    # Print BLEU scores
    bleu_overall = sum(bleu_scores) / len(bleu_scores)
    for idx, score in enumerate(bleu_scores):
        print(f"Sentence {idx + 1}: BLEU Score = {score}")


    metor_overall = sum(meteor_scores) / len(meteor_scores)
    for idx, score in enumerate(meteor_scores):
        print(f"Sentence {idx + 1}: Meteor Score = {score}")

    comet_overall = sum(comet_scores) / len(comet_scores)
    for idx, score in enumerate(comet_scores):
        print(f"Sentence {idx + 1}: Comet Score = {score}")

    print(f"Overall BLEU Score = {bleu_overall}")
    print(f"Overall Meteor Score = {metor_overall}")
    print(f"Overall Comet Score = {comet_overall}")

