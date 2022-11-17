import torch
import json, threading, argparse, datetime, platform
from kafka import KafkaConsumer, KafkaProducer
from model import BruceModel
from ctypes import *

if platform.system() == 'Windows':
    CDLL("C:\\Users\\BruceNguyen\\miniconda3\\Lib\\site-packages\\confluent_kafka.libs\\librdkafka-5d2e2910.dll")

from confluent_kafka import SerializingProducer, DeserializingConsumer
from confluent_kafka.serialization import StringSerializer, StringDeserializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer


model = BruceModel.load_from_checkpoint('./model_checkpoint/LSTM.ckpt')
if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'
model.eval()


class PigData(object):
    def __init__(self, inputs, target, time):
        self.inputs = inputs
        self.target = target
        self.time = time


class PigPrediction(object):
    def __init__(self, target, prediction, time):
        self.target = target
        self.prediction = prediction
        self.time = time


def data_to_dict(obj, ctx):
    return PigData(obj['inputs'], obj['target'], obj['time'])


def prediction_to_dict(obj: PigPrediction, ctx):
    return obj.__dict__


def predict_inputs(inputs):
    with torch.no_grad():
        cls_out, dt_out, id_out = model(torch.FloatTensor(inputs).unsqueeze(0).to(device))
    cls_out = torch.sigmoid(cls_out.flatten()).cpu()
    # prediction = dt_out.flatten().item() if cls_out > 0.5 else 0
    prediction = dt_out.flatten().cpu().item()
    return prediction


def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed for User record {}: {}".format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.format(
        msg.key(), msg.topic(), msg.partition(), msg.offset()))


def consuming(args):
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

    avro_deserializer = AvroDeserializer(schema_registry_client,
                                         schema_str,
                                         data_to_dict)
    string_deserializer = StringDeserializer('utf_8')

    consumer_conf = {'bootstrap.servers': args.bootstrap_servers,
                     'key.deserializer': string_deserializer,
                     'value.deserializer': avro_deserializer,
                     'group.id': args.group,
                     'auto.offset.reset': "earliest"}

    consumer = DeserializingConsumer(consumer_conf)
    consumer.subscribe([topic])
    err = 0

    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None:
                continue

            data = msg.value()
            prediction = predict_inputs(data.inputs)
            update_prediction(msg.key(), data.target, prediction)
            if data is not None:
                print("User record {}\tTarget: {}\tPrediction: {}".format(msg.key(), round(data.target, 2), prediction))
                if prediction != data.target:
                    err += 1
                    print("Number of error: ", err)
        except KeyboardInterrupt:
            break
    consumer.close()


def update_prediction(request_id, target, prediction):
    topic = 'pig-predictions'

    schema_str = """
            {
                "namespace": "confluent.io.examples.serialization.avro",
                "name": "PigPrediction",
                "type": "record",
                "fields": [
                    {
                        "name": "target", 
                        "type": "float"
                    },
                    {
                        "name": "prediction", 
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
    # schema_registry_client.set_compatibility("pig-predictions-value", "NONE") # Update schema if needed

    avro_serializer = AvroSerializer(schema_registry_client,
                                     schema_str,
                                     prediction_to_dict)

    producer_conf = {'bootstrap.servers': args.bootstrap_servers,
                     'key.serializer': StringSerializer('utf_8'),
                     'value.serializer': avro_serializer}

    producer = SerializingProducer(producer_conf)

    producer.poll(0.0)
    mess_id = request_id
    date_time = datetime.datetime.now()
    data = PigPrediction(target, prediction, date_time)
    producer.produce(topic=topic,
                     key=mess_id,
                     value=data,
                     # on_delivery=delivery_report
                     )
    producer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='bootstrap_servers', default='localhost:9092', type=str, help='Kafka Host')
    parser.add_argument('-s', dest="schema_registry", default='localhost:8081', help="Schema Registry")
    parser.add_argument('-t', dest="topic", default='pig-push-data', help="Topic name")
    parser.add_argument('-g', dest="group", default="data-consuming1", help="Consumer group")
    args = parser.parse_args()

    consuming(args)
