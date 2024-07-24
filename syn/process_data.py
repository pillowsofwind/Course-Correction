import json
from itertools import combinations


with open('train_data.jsonl', 'r') as f:
    lines = f.readlines()


with open('pairwise_dataset.jsonl', 'w') as f1:

    for i in range(0, len(lines), 6):
        group = lines[i:i+6] 
        if len(group) < 6:
            print(i)
            break 

        for a, b in combinations(range(6), 2):
            prompt = json.loads(group[a])['prompt']
            chosen = json.loads(group[a])['response']
            reject = json.loads(group[b])['response']
            data = {"prompt": prompt, "chosen": chosen, "rejected": reject}
            f1.write(json.dumps(data) + '\n')
