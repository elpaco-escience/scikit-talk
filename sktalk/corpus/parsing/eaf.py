from speach import elan

class ChaFile(InputFile):
    eaf = elan.read_eaf(filename)
    for tier in eaf:
        for ann in tier:
            _df = _df.append(pd.DataFrame({'begin': ann.from_ts, 'end' : ann.to_ts, 'participant' : tier.ID, 'utterance' : ann.text, 'source': path+filename}, index=[0]), ignore_index=True)
