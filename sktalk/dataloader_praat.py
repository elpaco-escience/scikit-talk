
def praat_short_to_long(path, path_out):
    for filename in os.listdir(path):
        if filename.endswith(".TextGrid"):
            print("opening new file") ## delete this later
            grid = textgrids.TextGrid(filename)
            os.mkdir(''.join(path_out))
            grid.write(path_out+filename)
    print("all done")


# script to convert .TextGrid (praat-long) to .csv

#!/usr/bin/python
# textgridall2csv.py
# D. Gibbon
# 2016-03-15
# 2016-03-15 (V02, includes filename in CSV records)

#-------------------------------------------------
# Import standard Python modules

import os, sys, re

#-------------------------------------------------
# Text file input / output

def inputtextlines(filename):
    handle = open(filename,'r')
    linelist = handle.readlines()
    handle.close()
    return linelist

def outputtext(filename, text):
    handle = open(filename,'w')
    handle.write(text)
    handle.close()

#-------------------------------------------------
# Conversion routine

def converttextgrid2csv(textgridlines,textgridname):

    csvtext = ''
    for line in textgridlines[9:]:
        line = re.sub('\n','',line)
        line = re.sub('^ *','',line)
        linepair = line.split(' = ')
        if len(linepair) == 2:
            if linepair[0] == 'class':
                classname = linepair[1]
            if linepair[0] == 'name':
                tiername = linepair[1]
            if linepair[0] == 'xmin':
                xmin = linepair[1]
            if linepair[0] == 'xmax':
                xmax = linepair[1]
            if linepair[0] == 'text':
                text = linepair[1]
                diff = str(float(xmax)-float(xmin))
                csvtext += textgridname + '\t' + classname + '\t' + tiername + '\t' + text + '\t' + xmin + '\t' + xmax + '\t' + diff + '\n'
    return csvtext

#-------------------------------------------------
# Define CSV header for individual files

header = '# ' + sys.argv[0] + 'TextGrid to CSV (D. Gibbon, 2008-11-23)\n# Open the file with OpenOffice.org Calc or MS-Excel.\nName\tTierType\tTierName\tLabel\tStart\tEnd\tDuration\n'

#-------------------------------------------------
# Get filenames.
try:
    textgridfiles = sorted(os.listdir('.'))
except:
    print ("Problem with input files.")
    exit()

#-------------------------------------------------
# Check for TextGrid filenames.
try:
    textgridfiles = [x for x in textgridfiles if x.endswith('.TextGrid')]
    if textgridfiles == []:
        print ("No TextGrid files to process.")
        exit()
except:
    print ("File input problem.")
    exit()

#-------------------------------------------------
# Process one TextGrid file at a time.

try:

    allcsvs = ''
    for filename in textgridfiles:

# Create TextGrid name and CSV file name.
        tgname = re.sub('.TextGrid','',filename)
        csvname = re.sub('.TextGrid','.csv',filename)

# Conversion.
        print ('Converting',filename,'to',csvname)
        try:
            textgrid = inputtextlines(filename)
            csvtext = converttextgrid2csv(textgrid,tgname)
            if csvtext == '':
                print ('No data in file',filename)
                exit()
        except:
            print ("Problem with TextGrid data.")
            exit()

# Create separate CSV output for each TextGrid.
        csvheadedtext = header + csvtext
        outputtext(csvname,csvheadedtext)

# Put all CSVs into a single file.
        allcsvs += csvtext

    print ('Saving all csvs in one file: ALLCSVs.csv')
    outputtext('ALLCSVs.csv',allcsvs)

    print ("Done.")

except:
    print ("Unknown error.")

#--- EOF -----------------------------------------