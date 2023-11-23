import abc
import json
from pathlib import Path
import csv
import uuid



class Writer(abc.ABC):
    @abc.abstractmethod
    def asdict(self):
        return NotImplemented

    def write_json(self, path: str = "./file.json"):
        """
        Write an object to a JSON file.

        Args:
            name (str): The file name
            directory (str): The path to the directory where the .json
            file will be saved.
        """
        _path = Path(path).with_suffix(".json")

        object_dict = self.asdict()

        with open(_path, "w", encoding='utf-8') as file:
            json.dump(object_dict, file, indent=4)
    
    def write_csv(self, folder: str = "link/to/folder/"):
        """
        Write a conversation dictionary to three .csv files.
            conversation.csv: containing the metadata of the conversation
            participants.csv: containing metadata about the participants
            utterances.csv: containing the utterances of the conversation            
        """
        object_dict = self.asdict()
        unique_id = str(uuid.uuid4())
        
        # Conversation.csv
        headers_conversation = []
        headers_conversation.append('unique_id')

        for key in object_dict.keys():
            if key not in headers_conversation and key != 'Utterances' and key != 'Participants':
                headers_conversation.append(key)
        
        with open(folder+"conversation.csv", "w",  newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers_conversation)
            writer.writeheader()
            row = {header: object_dict.get(header, '') for header in headers_conversation}
            row['unique_id'] = unique_id
            writer.writerow(row)
        
        # participants.csv
        headers_participants = []
        headers_participants.append('unique_id')
        
        if 'Participants' in object_dict and object_dict['Participants']:
            first_participant_key = next(iter(object_dict['Participants']))
            first_participant_data = object_dict['Participants'][first_participant_key]

            for key in first_participant_data.keys():
                if key not in headers_participants:
                    headers_participants.append(key)

        headers_participants.insert(0,'participant')
        
        with open(folder+"participants.csv", 'w', newline='', encoding='utf-8') as file:
            headers_participants = ['unique_id'] + [header for header in headers_participants if header != 'unique_id']
            writer = csv.DictWriter(file, fieldnames=headers_participants)
            writer.writeheader()
            
            for participant_id, participant_details in object_dict['Participants'].items():
                row = {}                
                row['unique_id'] = unique_id
                
                if 'participant' in headers_participants:
                    row['participant'] = participant_id
                
                for key in participant_details:                    
                    if key in headers_participants:
                        row[key] = participant_details[key]               
                writer.writerow(row)
        # utterances.csv
        headers_utterances = []
        headers_utterances.append('unique_id')
        
        if 'Utterances' in object_dict and len(object_dict['Utterances']) > 0:
            for key in object_dict['Utterances'][0].keys():
                if key not in headers_utterances:
                    headers_utterances.append(key)
        
        with open(folder+'utterances.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers_utterances)
            writer.writeheader()

            for utterance in object_dict['Utterances']:                
                row = {}
                for header in headers_utterances:
                    if header in object_dict and header != 'Utterances':
                        if isinstance(object_dict[header], list):
                            row[header] = ', '.join(object_dict[header])
                        else:
                            row[header] = object_dict[header]
                    elif header in utterance:
                        row[header] = utterance[header]
                row['unique_id'] = unique_id 
                writer.writerow(row)
    
    def _write_csv(self, folder: str = "./"):
        #Conv
        with open(folder+"conversation.csv", "w",  newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers_conversation)
            writer.writeheader()
            row = {header: object_dict.get(header, '') for header in headers_conversation}
            row['unique_id'] = unique_id
            writer.writerow(row)
        #Part
        with open(folder+"participants.csv", 'w', newline='', encoding='utf-8') as file:
            headers_participants = ['unique_id'] + [header for header in headers_participants if header != 'unique_id']
            writer = csv.DictWriter(file, fieldnames=headers_participants)
            writer.writeheader()
            
            for participant_id, participant_details in object_dict['Participants'].items():
                row = {}                
                row['unique_id'] = unique_id
                
                if 'participant' in headers_participants:
                    row['participant'] = participant_id
                
                for key in participant_details:                    
                    if key in headers_participants:
                        row[key] = participant_details[key]               
                writer.writerow(row)

        #Utterances
         with open(folder+'utterances.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers_utterances)
            writer.writeheader()

            for utterance in object_dict['Utterances']:                
                row = {}
                for header in headers_utterances:
                    if header in object_dict and header != 'Utterances':
                        if isinstance(object_dict[header], list):
                            row[header] = ', '.join(object_dict[header])
                        else:
                            row[header] = object_dict[header]
                    elif header in utterance:
                        row[header] = utterance[header]
                row['unique_id'] = unique_id 
                writer.writerow(row)

        #Write header
        def _write_header(f, )