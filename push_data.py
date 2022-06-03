import pandas as pd, uuid, json, datetime, time, argparse, os
from train import get_data
from ctypes import *
CDLL("C:\\Users\\BruceNguyen\\miniconda3\\Lib\\site-packages\\confluent_kafka.libs\\librdkafka-5d2e2910.dll")

from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer


class PigSensor(object):
    def __init__(self, inputs, target, time):
        self.inputs = inputs
        self.target = target
        self.time = time


def data_to_dict(obj: PigSensor, ctx):
    return obj.__dict__


def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed for User record {}: {}".format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.format(
        msg.key(), msg.topic(), msg.partition(), msg.offset()))


def producing(args):
    train_inputs, train_cls_label, train_deposit_thickness, train_inner_diameter, _, _ = get_data('val_new.h5',
                                                                                                  False,
                                                                                                  True)
    df = pd.DataFrame({'inputs': train_inputs.tolist(), 'labels': train_deposit_thickness.flatten()})

    topic = args.topic

    schema_str = """
        {
            "namespace": "confluent.io.examples.serialization.avro",
            "name": "PigSensor",
            "type": "record",
            "fields": [
                {
                    "name": "inputs",
                    "type": {
                        "type": "array",
                        "items": "float",
                        "name": "input"
                    },
                    "default": []
                },
                {
                    "name": "target", 
                    "type": "float"
                },
                {
                    "name": "time",
                    "type": {
                        "type": "long",
                        "logicalType": "timestamp-millis"
                    }
                }
            ]
        }
        """

    schema_registry_conf = {'url': args.schema_registry}
    schema_registry_client = SchemaRegistryClient(schema_registry_conf)

    avro_serializer = AvroSerializer(schema_registry_client,
                                     schema_str,
                                     data_to_dict)

    producer_conf = {'bootstrap.servers': args.bootstrap_servers,
                     'key.serializer': StringSerializer('utf_8'),
                     'value.serializer': avro_serializer}

    producer = SerializingProducer(producer_conf)
    for i in range(100000):
        producer.poll(0.0)
        try:
            mess_id = str(uuid.uuid4())
            date_time = datetime.datetime.now()
            data = PigSensor(inputs=df.inputs[i], target=df.labels[i], time=date_time)
            producer.produce(topic=topic, key=mess_id, value=data,
                             on_delivery=delivery_report)
        except KeyboardInterrupt:
            producer.flush()
            break
        except ValueError:
            print("Invalid input, discarding record...")
            continue

        print(' -- PRODUCER: Sent message at {}, message id {}'.format(
            date_time.strftime('%Y-%m-%d %H:%M:%S'),
            mess_id)
        )
        time.sleep(1)

    producer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='bootstrap_servers', default='localhost:9092', type=str, help='Kafka Host')
    parser.add_argument('-s', dest="schema_registry", default='localhost:8081', help="Schema Registry")
    parser.add_argument('-t', dest="topic", default='pig-push-data', help="Topic name")
    args = parser.parse_args()

    producing(args)
