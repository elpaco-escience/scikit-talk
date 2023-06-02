from .parser import Parser
import pylangacq

class ChaParser(Parser):
  def parse(self, file):
    chatfile = pylangacq.read_chat(file)
    chat_utterances = chatfile.utterances(by_files=False)
    for row in chat_utterances:
      utterance_dict = []
      utterance_dict['speaker'] = row.participant
      utterance_dict['time'] = str(row.time_marks)
      utterance_dict['utterance'] = str(row.tiers)
      utterance_dict['source'] = file