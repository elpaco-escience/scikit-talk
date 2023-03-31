corpora_list = []

metadatafile = pd.read_csv("_overview.csv", encoding="ISO-8859-1", sep=',')
corpora_for_d_latest = pd.DataFrame(columns=['corpus_path','langshort','langfull'])
# loop over csv files (language corpus)
for index, row in metadatafile.iterrows():
        if row["ElPaCo_included"] == "yes":
            corpus_name = row["File_name"]
            corpus_path = './Elpaco dataset/'+corpus_name+'.csv'
            langshort = row["langshort"]
            langfull = row["Langfull"]
            corpora_list.append([corpus_path, langshort,langfull])

corpora_for_d_latest = pd.DataFrame(corpora_list, columns = ['language', 'langshort', 'langfull'])

print(corpora_for_d_latest)