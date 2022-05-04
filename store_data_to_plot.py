import torch
import json, threading, pandas as pd
from kafka import KafkaConsumer, KafkaProducer

x = []
targets = []
predicts = []


def consuming():
    c = KafkaConsumer('pig_predictions', bootstrap_servers='10.8.8.105:9092', group_id='predict_consumer')
    for msg in c:
        mess = json.loads(msg.value)
        x.append(mess['time'])
        targets.append(mess['target'])
        predicts.append(mess['prediction'])

        if len(x) >= 60:
            x.pop(0)
            targets.pop(0)
            predicts.pop(0)

        df_plot = pd.DataFrame({'idx': x, 'target': targets, 'predict': predicts})
        df_plot.to_csv('df_to_plot.csv', index=False)
        
    c.close()


if __name__ == '__main__':
    consuming()