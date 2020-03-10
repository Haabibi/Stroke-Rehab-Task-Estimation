import csv
def parse_verb():
    class_dict = {}
    with open('../annotations/EPIC_verb_classes.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == 'verb_id': pass
            else: class_dict[int(row[0])] = row[1].strip()

    return class_dict

def parse_noun():
    class_dict = {}
    with open('../annotations/EPIC_noun_classes.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == 'noun_id': pass
            else: class_dict[int(row[0])] = row[1].strip()

    return class_dict

if __name__=='__main__':
    print(parse_noun())
