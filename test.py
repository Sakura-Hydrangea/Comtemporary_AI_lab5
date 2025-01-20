import torch
from tqdm import tqdm
import pandas as pd
from dataprocessing import testloader


class Test:
    def __init__(self, args):
        self.args = args

    def test(self, model, state_dict_path):
        print('\ntest start')
        self.test_loader = testloader(self.args)
        model.load_state_dict(torch.load(state_dict_path))
        outputs = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(self.test_loader, desc='test'):
                text_ids, attention_masks, imgs = batch[0].to(self.args.device), batch[1].to(self.args.device), batch[2].to(self.args.device)
                output = model(text_ids, attention_masks, imgs)
                outputs.extend(output)
            self.saveres(outputs)

    def saveres(self, outputs):
        pred = [output.argmax() for output in outputs]
        label2idx = {'positive': 0, 'neutral': 1, 'negative': 2}
        idx2label = {v: k for k, v in label2idx.items()}
        pred = [idx2label[i.item()] for i in pred]
        pd_data = pd.read_csv('data/test_without_label.txt')
        pd_data['tag'] = pred
        pd_data.to_csv(f'result.txt', index=False)
