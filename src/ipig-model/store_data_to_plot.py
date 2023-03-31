import argparse
import json, threading, pandas as pd
from kafka import KafkaConsumer, KafkaProducer

x = []
targets = []
predicts = []


def consuming(args):
    c = KafkaConsumer(args.topic, bootstrap_servers=args., group_id='predict_consumer')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='bootstrap_servers', default='localhost:9092', type=str, help='Kafka Host')
    parser.add_argument('-s', dest="schema_registry", default='localhost:8081', help="Schema Registry")
    parser.add_argument('-t', dest="topic", default='pig-predictions', help="Topic name")
    parser.add_argument('-g', dest="group", default='predict_consumer', help="Consumer group")
    args = parser.parse_args()

    consuming(args)