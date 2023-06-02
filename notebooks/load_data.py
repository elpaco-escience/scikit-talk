from sktalk.corpus.parsing.cha import ChaParser

if __name__ == "__main__":
    french = ChaParser().parse("../data/french1/5136.cha")
    print(french.utterances)

