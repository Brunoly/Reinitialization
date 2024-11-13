import os
import json

def print_final_test_acc(folder_path):
    for root, dirs, files in os.walk(folder_path):
        
        for file in files:
            if file.endswith('_history.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'history' in data and 'test_acc' in data['history']:
                        final_test_acc = data['history']['test_acc'][-1]
                        config = data.get('config', {})
                        import pprint
                        pprint.pprint(f"{config}, Final Test Accuracy: {final_test_acc}")



folder_path = 'Reinitialization/modelos'

print_final_test_acc(folder_path)