import torch
import json, threading, argparse
from kafka import KafkaConsumer, KafkaProducer
from model import BruceModel

device = 'gpu:0' if torch.cuda.is_available() else 'cpu'
model = BruceModel.load_from_checkpoint('./model_checkpoint/LSTM.ckpt', map_location=device)
model.eval()


def predict(inputs):
    with torch.no_grad():
        cls_out, dt_out, id_out = model(torch.FloatTensor(inputs).unsqueeze(0))
    cls_out = torch.sigmoid(cls_out.flatten())
    prediction = dt_out.flatten().item() if cls_out > 0.5 else 0
    return prediction


def consuming():
    c = KafkaConsumer('pig-push-data', bootstrap_servers=KAFKA_HOST, group_id='data_consumer')
    for msg in c:
        mess = json.loads(msg.value)
        print('** CONSUMER: Received request id {}'.format(mess['request_id']))
        prediction = predict(mess['input'])
        update_prediction(mess['request_id'], mess['target'], prediction, mess['time'])
    c.close()


def update_prediction(request_id, target, prediction, time):
    p = KafkaProducer(bootstrap_servers=KAFKA_HOST)
    mess = {
        'request_id': request_id,
        'target': target,
        'prediction': prediction,
        'time': time,
    }
    p.send('pig-predictions', json.dumps(mess).encode('utf-8'))
    print('-- PRODUCER: Predict request id {} as {}, the true label is {}'.format(mess['request_id'],
                                                                                  mess['prediction'],
                                                                                  mess['target']))
    p.flush()


if __name__ == '__main__':
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--host', default='localhost:9092', type=str, help='Kafka Host')
    args = model_parser.parse_args()
    KAFKA_HOST = args.host

    consuming()
