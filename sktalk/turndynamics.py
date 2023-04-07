import os
import glob
import math
import re
import datetime
import grapheme
import numpy as np
import pandas as pd
from collections import Counter

from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed


def readcorpus(filename, langshort=None, langfull=None):
    """ returns a formatted language corpus with turn and transition measures

    :param a: filename of the corpus
    :type a: string
    :param b: short version of language name, defaults to none
    :type b: string, optional
    :param c: full version of language name, defaults to none
    :type c: string, optional

    :return: formatted dataframe of the language corpus
    """
    # convert time strings to the ISO 8601 time format hh:mm:ss.sss
    def _converttime(text):
        if pd.isna(text) == True:
            return pd.NA
        else:
            h, m, s = text.split(':')
            return int(datetime.timedelta(hours=int(h),
                                          minutes=int(m),
                                          seconds=float(s)).total_seconds()*1000)

    # number of unique sources
    def _getsourceindex(source):
        return n_sources.index(source) + 1

    # talk, laugh, breath, or other conduct classification
    def _getnature(utterance):
        if pd.isna(utterance) == True:
            return pd.NA
        if utterance == '[laugh]':
            return 'laugh'
        if utterance == '[breath]':
            return 'breath'
        if utterance in ['[cough]', '[sneeze]', '[nod]', '[blow]', '[sigh]',
                         '[yawn]', '[sniff]', '[clearsthroat]',
                         '[lipsmack]', '[inhales]', '[groan]']:
            return utterance
        else:
            return 'talk'

    # count number of characters
    def _getnchar(utterance):
        if pd.isna(utterance) == True:
            return pd.NA
        else:
            utterance = Counter(utterance.replace(" ", ""))
            return sum(utterance.values())

    # create a 'window' for each utterance
    # The window looks at 10s prior the begin of the current utterance (lookback)
    # Only turns that begin within this lookback are included
    # in the window. This means that if the prior turn began later
    # than 10s before the current utterance, then the prior turn is
    # not included in the window.
    def _createwindow(begin, participant):
        lookback = 10000
        lookfwd = 0
        filter = (df_transitions['begin'] >= (
            begin - lookback)) & (df_transitions['begin'] <= (begin + lookfwd))
        window = df_transitions.loc[filter]
        # identify who produced the utterance
        window['turnby'] = np.where(window['participant'] == participant, 'self',
                                    'other')
        # calculate duration of all turns in window
        stretch = window['end'].max() - window['begin'].min()
        # calculate sum of all turn durations
        talk_all = window['duration'].sum()
        # calculate amount of talk produced by the participant in relation
        # to the total amount of talk in the window
        try:
            talk_rel = window.loc[window['turnby'] ==
                                  'self']['duration'].sum() / talk_all
        except ZeroDivisionError:
            talk_rel = pd.NA
        # calculate amount of loading of the channel
        # (1 = no empty space > overlap, < silences)
        load = talk_all / stretch
        # calculate total amount of turns in this time window
        turns_all = len(window.index)
        # calculate amount of turns by this participant relative to turns by others
        try:
            turns_rel = (
                len(window[window['turnby'] == 'self'].index)) / turns_all
        except ZeroDivisionError:
            turns_rel = pd.NA

        participants = window['participant'].nunique()
        # create list of all measures computed
        measures = [talk_all, talk_rel, load,
                    turns_all, turns_rel, participants]
        return measures

    df = pd.read_csv(filename)
    filename = re.sub('.csv', "", filename)
    filename = re.sub('\.\/ElPaCo Dataset\/', '', filename)
    # filename = re.sub('ElPaCo dataset\/', '', filename)
    df['language'] = re.sub("[0-9]", "", filename)
    if langshort is not None:
        df['langshort'] = langshort
    else:
        df['langshort'] = df['language']
    if langfull is not None:
        df['langfull'] = langfull
    else:
        df['langfull'] = df['language']
    df['corpus'] = filename
    df['begin'] = df['begin'].apply(_converttime)
    df['end'] = df['end'].apply(_converttime)

    # calculate duration of the turn
    df['duration'] = df['end'] - df['begin']

    # define improbably long (more than 40 seconds) and negative durations
    n_weird_durations = df.loc[(
        (df['duration'] > 40000) | (df['duration'] < 0))]

    # set weird durations to NA under the ff columns: begin, end, and duration
    df.loc[(df['duration'] > 40000) | (
        df['duration'] < 0), ['duration']] = pd.NA
    df.loc[(df['duration'] > 40000) | (df['duration'] < 0), ['end']] = pd.NA
    df.loc[(df['duration'] > 40000) | (df['duration'] < 0), ['begin']] = pd.NA

    # create UID

    # list of unique sources in the corpus
    n_sources = df['source'].unique().tolist()
    # length of the number of sources (i.e. 20 sources = 2 chars), for padding
    x = len(str(len(n_sources)))
    # length of the number of turns in a source
    # (i.e. 100 conversations = 3 chars), for padding
    y = len(str(len(df.groupby(['source', 'utterance']).size())))

    # UID format: language-source number-turn number (within a source)
    uidbegin = np.where(pd.isna(df['begin']) ==
                        True, 'NA', df['begin'].astype(str))
    df['uid'] = df['language'] + '-' + (df['source'].apply(_getsourceindex)).astype(str).str.zfill(
        x) + '-' + (df.groupby(['source']).cumcount() + 1).astype(str).str.zfill(y) + '-' + uidbegin

    # deal with "unknown utterance" content
    na_strings = ['[unk_utterance', '[unk_noise]', '[distortion]',
                  '[background]', '[background] M', '[static]', 'untranscribed',
                  '[noise]', '[inintel]', '[distorted]', 'tlyam kanəw']

    # set unknown utterances to NA
    df.loc[(df['utterance'].isin(na_strings)), ['utterance']] = pd.NA
    n_unknown = df['utterance'][df['utterance'].isin(na_strings)].count()

    # get nature of utterance
    df['nature'] = df['utterance'].apply(_getnature)

    # create a stripped version of the utterance
    df['utterance_stripped'] = df['utterance'].str.strip()
    df['utterance_stripped'] = df['utterance_stripped'].str.replace(r'\[[^[]*\]',
                                                                    '', regex=True)
    df['utterance_stripped'] = df['utterance_stripped'].str.replace(r'[\\(\\)]+',
                                                                    '', regex=True)
    # set blank utterances to NA
    df.loc[df['utterance_stripped'] == '', 'utterance_stripped'] = pd.NA

    # measure number of words by counting spaces
    df['nwords'] = df['utterance_stripped'].str.count(' ') + 1

    # measure number of characters
    df['nchar'] = df['utterance_stripped'].apply(_getnchar)  # .astype(float)

    # add turn and frequency rank measures

    # create a new dataframe without NA utterances (for easier calculations)
    df_ranking = df.dropna(subset=['utterance_stripped'])
    # count how frequent the utterance occurs in the corpus
    df_ranking['n'] = df_ranking.groupby(
        'utterance')['utterance'].transform('count').astype(float)
    # rank the frequency of the utterance
    df_ranking['rank'] = df_ranking['n'].rank(method='dense', ascending=False)
    # calculate total number of uttrances
    df_ranking['total'] = df_ranking['n'].sum()
    # calculate frequency of utterance in relation to the total number of utterances
    df_ranking['frequency'] = df_ranking['n'] / df_ranking['total']
    # merge the new dataframe with the original dataframe
    df = pd.merge(df, df_ranking)

    # categorize overlap, look at overlap with turns up to four positions down
    # overlap can either be full or partial
    # set to NA if no overlap is found
    df['overlap'] = np.where((df['begin'] > df['begin'].shift(1)) & (df['end'] < df['end'].shift(1)) |
                             (df['begin'] > df['begin'].shift(2)) & (df['end'] < df['end'].shift(2)) |
                             (df['begin'] > df['begin'].shift(3)) & (df['end'] < df['end'].shift(3)) |
                             (df['begin'] > df['begin'].shift(4)) & (
        df['end'] < df['end'].shift(4)),
        'full', np.where((df['begin'] > df['begin'].shift()) & (df['begin'] <= df['end'].shift()),
                         'partial', pd.NA))

    # identify who produced the prior utterance: other, self,
    # or self during other (if previous utterance by the same participant
    # was fully overlapped by an utterance of a different pariticpant)
    # the priorby of the first utterance in the corpus is set to NA
    df['priorby'] = np.where(df['participant'].index == 0, pd.NA,
                             np.where(df['participant'] != df['participant'].shift(),
                                      'other', np.where((df['overlap'].shift() == 'full') &
                                                        (df['participant'].shift(
                                                        ) == df['participant']),
                                                        'self_during_other', 'self'
                                                        )))

    # calculate FTO (Flow Time Overlap)
    # This refers to the duration of the overlap between the current utterance
    # and the most relevant prior turn by other, which is not necessatily the
    # prior row in the df. By default we only get 0, 1 and 5 right. Cases 2
    # and 3 are covered by a rule that looks at turns coming in early for which
    # prior turn is by self but T-2 is by other. Some cases of 4 (but not all is
    # covered by looking for turns that do not come in early but have a prior
    # turn in overlap and look for the turn at T-2 by a different participant.

    # A turn doesn't receive an FTO if it follows a row in the db that doesn't
    # have timing information, or if it is such a row.

    # A [------------------]      [0--]
    # B      [1-]  [2--] [3--] [4--]    [5--]

    df['FTO'] = np.where((df['priorby'] == 'other') & (df['begin'] - df['begin'].shift() < 200) &
                         (df['priorby'].shift() != 'other'), df['begin'] -
                         df['end'].shift(2),
                         np.where((df['priorby'] == 'other') &
                                  (df['begin'] - df['begin'].shift() < 200) &
                                  (df['priorby'].shift() != 'self') &
                                  df['priorby'].shift(2) == 'other',
                                  df['begin'] - df['end'].shift(3),
                                  np.where((df['priorby'] == 'self_during_other') &
                                           (df['participant'].shift(
                                               2) != df['participant']),
                                           df['begin'] - df['end'].shift(2),
                                           np.where((df['priorby'] == 'self_during_other') &
                                                    (df['priorby'].shift()
                                                     == 'self_during_other'),
                                                    df['begin'] -
                                                    df['end'].shift(3),
                                                    np.where(df['priorby'] == 'other',
                                                             df['begin'] -
                                                             df['end'].shift(),
                                                             np.where(df['priorby'] == 'self', pd.NA, pd.NA
                                                                      ))))))

    # identify whther a turn is overlapped by what succeeds it
    # if not, set to NA
    df['overlapped'] = np.where((df['begin'] < df['begin'].shift(-1)) &
                                (df['end'] > df['begin'].shift(-1)), 'overlapped', pd.NA)

    # set FTO to NA if it is higher than 10s or lower than -10s, on the
    # grounds that (a) psycholinguistically it is implausible that these
    # actually relate to the end of the 'prior', and (b) conversation
    # analytically it is necessary to treat such cases on their
    # own terms rather than take an FTO at face value

    df['FTO'] = np.where(df['FTO'] > 9999, pd.NA, np.where(
        df['FTO'] < -9999, pd.NA, df['FTO']))
    # set FTO to NA if it is negative < -99999, on the
    # grounds that (a) psycholinguistically it is
    # impossible to relate to the end of the 'prior' turn,
    # and (b) conversation analytically it is necessary
    # to treat such cases on their own terms rather than
    # take an FTO at face value

    # add transitions metadata

    # create new dataframe with only the relevant columns
    df_transitions = df.copy()
    df_transitions = df_transitions.drop(columns=['langshort', 'langfull',
                                                  'corpus', 'nature',
                                                  'utterance_stripped',
                                                  'nwords', 'nchar', 'n',
                                                  'rank', 'total',
                                                  'frequency', 'overlap'])

    # put all the calculated transition measures into one column
    df['transitions'] = df.apply(lambda x: _createwindow(x['begin'],
                                                         x['participant']),
                                 axis=1)

    # split the list into six columns, one column representing each measure
    df_split = pd.DataFrame(df['transitions'].tolist(), columns=['talk_all', 'talk_rel', 'load',
                                                                 'turns_all', 'turns_rel', 'participants'])

    # add transition measures to original df
    df = pd.concat([df, df_split], axis=1)
    # drop column containing list of transition measures
    df = df.drop(columns='transitions')

    return df
