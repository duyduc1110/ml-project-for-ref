import pandas as pd, uuid, json, datetime, time
from kafka import KafkaProducer
from train import get_data


def producing():
    train_inputs, train_cls_label, train_deposit_thickness, train_inner_diameter, _, _ = get_data('train_new.h5',
                                                                                                  False,
                                                                                                  True)
    df = pd.DataFrame({'inputs': train_inputs.tolist(), 'labels': train_deposit_thickness.flatten()})

    producer = KafkaProducer(bootstrap_servers='10.8.8.105:9092')
    for i in range(100):
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
    producing()