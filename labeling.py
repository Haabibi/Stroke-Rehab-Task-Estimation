import csv 

csv_file = '/home/abigail/Downloads/EPIC_train_object_labels.csv'
f = open('/home/abigail/Downloads/EPIC_labeling_python.csv', "a")
obj = ['apple', 
        ['bowl', 'container', 'plate'], 
        ['cereal', 'shreddies', 'granola', 'cereal box'], 
        'brush', 
        'towel', 
        'knife', 
        'marker', 
        ['cup', 'mug', 'glass'], 
        'book', 
        ['pitcher', 'jug', 'carafe', 'kettle'],
        'plate',
        'scissor',
        ['shampoo' 'washing liquid', 'cleaning liquid', 'detergent', 'degreaser', 'cleanser', 'dish washing', 'cleaner'],
        ['can', 'tin', 'tomato tin'],
        'sponge',
        ['spoon', 'fork'],
        'tomato',
        'brush',
        'toothpaste',
        'bottle'
      ]

row_cnt = 0
idx = 0


for i in range(len(obj)):
    if type(obj[i]) == str: 
        noun = obj[i] 
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                noun_class, full_noun, p_id, v_id, frame, bbox = row
                if len(bbox) == 2: 
                    continue
                if noun in full_noun: 
                    f.write(str(i+1) + ', ' + noun + ', ' + p_id + ', '+ v_id + ', ' + frame + ', ' + bbox + '\n')
    else: 
        for n in obj[i]:
            noun = n 
            with open(csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader: 
                    noun_class, full_noun, p_id, v_id, frame, bbox = row
                    if len(bbox) == 2: continue
                    if noun in full_noun: 
                        f.write(str(i+1) + ', ' + noun + ', ' + p_id + ', '+ v_id + ', ' + frame + ', ' + bbox + '\n')

    """
    for row in reader:
        row_cnt += 1
        if row_cnt == 1:
            continue
        else:
            noun_class, noun, p_id, v_id, frame, bbox = row
            tmp_bbox = bbox.strip('][()').split(', ')
            if len(tmp_bbox) == 4:
                img_name = p_id + '/' + v_id + '/' + frame
                print(img_name)
                break
                x, y, w, h = tmp_bbox[0], tmp_bbox[1], tmp_bbox[2], tmp_bbox[3]
    """
f.close()
