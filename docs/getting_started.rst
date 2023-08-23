.. _getting_started:

Getting started
---------------

To get started with scikit-talk, you need a transcription file.
For example, you can download a file from the
`Griffith Corpus of Spoken Australian English <https://ca.talkbank.org/data-orig/GCSAusE/01.cha>`_.

Load a corpus using the parser for .cha files:

.. code-block:: python

    from sktalk.corpus.parsing.cha import ChaFile
    parsed_cha = ChaFile(download_file).parse()

Access the metadata with:

.. code-block:: python

    parsed_cha.metadata

Access the utterances with:

.. code-block:: python

    parsed_cha.utterances



