!pip install praat-textgrids

import os
import sys
import textgrids

import xml.etree.ElementTree as et
from collections import defaultdict
import pandas as pd
import os

import os
import csv
import regex as re
import numpy as np
import pandas as pd
!pip install pylangacq
import pylangacq
!pip install speach
from speach import elan
import datetime



def eaf_to_df(path, *path_out):
    _df = pd.DataFrame([])
    for filename in os.listdir(path):
        print("done, opening new file") ## delete this later
        if filename.endswith(".eaf"):
            eaf = elan.read_eaf(filename)
            for tier in eaf:
              for ann in tier:
                   _df = _df.append(pd.DataFrame({'begin': ann.from_ts, 'end' : ann.to_ts, 'participant' : tier.ID, 'utterance' : ann.text, 'source': path+filename}, index=[0]), ignore_index=True)
        else:
            continue
    if path_out:
      os.mkdir(''.join(path_out))
      _df.to_csv(''.join(path_out)+'data_eaf.csv' , index=False)
    # return _df.sort_values(['source', 'begin','end'], ascending=[True, True, False], inplace=True)
    data_eaf = _df
    return data_eaf

def cha_to_df(path, *path_out):
  #path = '/content/drive/MyDrive/SpokenBNC/testing/chat/new/'
  _df = pd.DataFrame()
  #read chat files one by one in a for loop, and append the conversations line by line to a dataframe

  for filename in os.listdir(path):
      if filename.endswith(".cha"):
          chatfile = pylangacq.read_chat(filename)
          chat_utterances=chatfile.utterances(by_files=False)
          for row in chat_utterances:
            _df = _df.append(pd.DataFrame({'speaker':row.participant, 'time':str(row.time_marks), 'utterance':str(row.tiers), 'source': path+filename}, index=[0]), ignore_index=True)
      else:
          continue

  # split time markers
  _df[['begin','end']] = _df.time.str.split(r', ', 1, expand=True)
  #do some reordering
  _df = _df[['begin', 'end', 'speaker', 'utterance', 'source']]
  # do some cleaning
  _df['begin'] = [re.sub(r'\(', "", str(x)) for x in _df['begin']]
  _df['end'] = [re.sub(r'\)', "", str(x)) for x in _df['end']]
  _df['utterance'] = [re.sub(r'^([^:]+):', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'^\s+', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'\s+$', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'\}$', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'^\"', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'\"$', "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r"^\'", "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r"\'$", "", str(x)) for x in _df['utterance']]
  _df['utterance'] = [re.sub(r'\\x15\d+_\d+\\x15', "", str(x)) for x in _df['utterance']]
  # filling up empty cells with NaN
  _df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
  _df.replace(r'None', np.nan, regex=True, inplace=True)
  _df.fillna(value=np.nan, inplace=True)
  # formatting timestamp
  _df['begin'] = pd.to_numeric(_df['begin'])
  _df['begin'] = pd.to_datetime(_df['begin'], unit='ms', errors='ignore')
  _df['begin'] = pd.to_datetime(_df['begin']).dt.strftime("%H:%M:%S.%f")#[:-3]
  _df['end'] = pd.to_numeric(_df['end'])
  _df['end'] = pd.to_datetime(_df['end'], unit='ms', errors='ignore')
  _df['end'] = pd.to_datetime(_df['end']).dt.strftime("%H:%M:%S.%f")#[:-3]

  if path_out:
    os.makedirs(path_out)
    _df.to_csv(''.join(path_out)+'data_cha.csv', index=False)

  data_cha = _df
  return data_cha

def ldc_to_df(path, *path_out):
    # define in path
    path=r'/content/'
    # create an empty dataframe
    _df = pd.DataFrame()
    # read the txts, turn them into dataframes and append
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
          textfile = pd.DataFrame()
          textfile[['time_speaker', 'utterance']] = pd.read_table(filename, sep=':', header=None , error_bad_lines=False)
          textfile['source'] = path+filename
          _df = _df.append(textfile)
        else:
          continue
    #filter out metadata
    meta_pattern = '^\#'
    filter = _df['time_speaker'].str.contains(meta_pattern)
    _df = _df[~filter]
    #reset index
    _df.reset_index(inplace=True, drop=True)
    # split speaker and time markers
    _df[['time','speaker']] = _df.time_speaker.str.split(r'\s(?!.*\s)', 1, expand=True)
    # split time markers
    _df[['begin','end']] = _df.time.str.split(r' ', 1, expand=True)
    # drop extra columns
    _df.drop(columns=['time_speaker', 'time'], inplace=True)
    # reorder
    _df = _df[['begin',  'end', 'speaker', 'utterance', 'source']]
    # formatting timestamp
    _df['begin'] = pd.to_numeric(_df['begin'], errors='coerce')
    _df['begin'] = pd.to_datetime(_df['begin'], unit='s', errors='ignore')
    _df['begin'] = pd.to_datetime(_df['begin']).dt.strftime("%H:%M:%S.%f")#[:-3]
    _df['end'] = pd.to_numeric(_df['end'], errors='coerce')
    _df['end'] = pd.to_datetime(_df['end'], unit='s', errors='ignore')
    _df['end'] = pd.to_datetime(_df['end']).dt.strftime("%H:%M:%S.%f")#[:-3]

    if path_out:
      _df.to_csv('/content/data_ldc.csv', index=False)

    data_ldc = _df
    return data_ldc


def flatten_xml(node, key_prefix=()):
    """
    Walk an XML node, generating tuples of key parts and values.
    """

    # Copy tag content if any
    text = (node.text or '').strip()
    if text:
        yield key_prefix, text

    # Copy attributes
    for attr, value in node.items():
        yield key_prefix + (attr,), value

    # Recurse into children
    for child in node:
        yield from flatten_xml(child, key_prefix + (child.tag,))


def dictify_key_pairs(pairs, key_sep='-'):
    """
    Dictify key pairs from flatten_xml, taking care of duplicate keys.
    """
    out = {}

    # Group by candidate key.
    key_map = defaultdict(list)
    for key_parts, value in pairs:
        key_map[key_sep.join(key_parts)].append(value)


    # Figure out the final dict with suffixes if required.
    for key, values in key_map.items():
        if len(values) == 1:  # No need to suffix keys.
            out[key] = values[0]
        else:  # More than one value for this key.
            for suffix, value in enumerate(values, 1):
                out[f'{key}{key_sep}{suffix}'] = value

    return out

def xml_to_df(path, *path_out):
    _df = pd.DataFrame([])
    for filename in os.listdir(path):
        print("done, opening new file") ## delete this later
        if filename.endswith(".skp"):
            tree = et.parse(filename)
            root = tree.getroot()
            rows = [dictify_key_pairs(flatten_xml(row)) for row in root]
            _df = _df.append(rows)

        else:
            continue
    if path_out:
      os.mkdir(''.join(path_out))
      _df.to_csv(''.join(path_out)+'data_eaf.csv' , index=False)
    # return _df.sort_values(['source', 'begin','end'], ascending=[True, True, False], inplace=True)
    data_eaf = _df
    return data_eaf

def transform_xml(xml_doc):
    attr = xml_doc.attrib
    for xml in xml_doc.iter('tau'):
        dict = attr.copy()
        dict.update(xml.attrib)

        yield dict

def transform_xml2(xml_doc):
    attr = xml_doc.attrib

    for xml in xml_doc.iter('tw'):
        dict = attr.copy()
        dict.update(xml.attrib)
        yield dict


def xml_to_df(path, *path_out):
    for filename in os.listdir(path):

        if filename.endswith(".skp"):
            print("done, opening new file", filename) ## delete this later
            etree =et.parse(filename)
            eroot = etree.getroot()
            trans = transform_xml(eroot)
            trans2 = transform_xml2(eroot)

            df2 = pd.DataFrame(list(trans))
            df = pd.DataFrame(list(trans2))

            df['ref'].replace("\.[^.]*$","",regex=True,inplace=True)
            df['spk'] = df.ref.map(dict(df2[['ref', 's']].values))

            df['utterance'] = df.groupby(['tb','tb', 'spk'])['w'].transform(lambda x: ' '.join(x))
            df = df.drop_duplicates(subset=['tb', 'tb', 'spk', 'utterance'], keep='first')

            df.drop(columns=['tt', 'tq', 'w'], inplace=True)

            df.rename(columns={'ref':'source', 'tb':'begin', 'te':'end','spk':'participant'}, inplace=True)
            df = df[['begin', 'end', 'participant','utterance','source',]]

            df['begin'] = pd.to_numeric(df['begin'])
            df['begin'] = pd.to_datetime(df['begin'], unit='s', errors='ignore')
            df['begin'] = pd.to_datetime(df['begin']).dt.strftime("%H:%M:%S.%f")#[:-3]
            df['end'] = pd.to_numeric(df['end'])
            df['end'] = pd.to_datetime(df['end'], unit='s', errors='ignore')
            df['end'] = pd.to_datetime(df['end']).dt.strftime("%H:%M:%S.%f")#[:-3]

            df.to_csv( ''.join(filename)+'.csv', index=False)
            print(filename, 'done one')