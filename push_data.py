import pandas as pd, uuid, json, datetime, time, argparse
from kafka import KafkaProducer
from train import get_data


def producing():
    train_inputs, train_cls_label, train_deposit_thickness, train_inner_diameter, _, _ = get_data('val_new.h5',
                                                                                                  False,
                                                                                                  True)
    df = pd.DataFrame({'inputs': train_inputs.tolist(), 'labels': train_deposit_thickness.flatten()})

    producer = KafkaProducer(bootstrap_servers=KAFKA_HOST)
    for i in range(1000):
        mess_id = str(uuid.uuid4())
        date_time = datetime.datetime.now()
        mess = {
            'request_id': mess_id,
            'input': df.inputs[i],
            'target': df.labels[i],
            'time': date_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        producer.send('pig_push_data', json.dumps(mess).encode('utf-8'))
        producer.flush()

        print(' -- PRODUCER: Sent message at {}, message id {}, with labels {} '.format(
            mess['target'],
            date_time.strftime('%Y-%m-%d %H:%M:%S'),
            mess_id)
        )
        time.sleep(1)


if __name__ == '__main__':
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--host', default='localhost:9092', type=str, help='Kafka Host')
    args = model_parser.parse_args()

    KAFKA_HOST = args.host
    producing()