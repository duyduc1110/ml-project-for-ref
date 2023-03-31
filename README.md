# IPIG Dashboard
This first section is about setting up everything in infrastructure to have event distribution system, database, pipeline ready

## Install Docker

```
sudo apt-get update

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
	
sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

## Setup Docker Compose file

```
volumes:
  - broker-data:/var/lib/kafka/data
  - broker-secrets:/etc/kafka/secrets
  - kafka-connect-data:/var/lib/kafka/data
  - kafka-connect-jars:/etc/kafka-connect/jars
  - kafka-connect-connect-secrets:/etc/kafka-connect/secrets
  - kafka-connect-secrets:/etc/kafka/secrets
  - schema-registry-secrets:/etc/schema-registry/secrets
  - zookeeper-data:/var/lib/zookeeper/data
  - zookeeper-log:/var/lib/zookeeper/log
  - zookeeper-secrets:/etc/zookeeper/secrets
  - postgres-data:/var/lib/postgresql/data
```
 
## Setup JBDC Sink Connectors to Posgres:
 
Access KSQL first 
```
sudo docker exec -it ksqldb ksql http://localhost:8088
```
 
Remember to push and predict some data first, therefore we have Kafka topics with AVRO schema already
 
Create connectors with Postgres:
```
CREATE SINK CONNECTOR SINK_PREDICTIONS WITH (
    'connector.class'                     = 'io.confluent.connect.jdbc.JdbcSinkConnector',
    'connection.url'                      = 'jdbc:postgresql://postgres:5432/',
    'connection.user'                     = 'postgres',
    'connection.password'                 = 'postgres',
    'topics'                              = 'pig-predictions',
    'key.converter'                       = 'org.apache.kafka.connect.storage.StringConverter',
    'value.converter'                     = 'io.confluent.connect.avro.AvroConverter',
    'value.converter.schema.registry.url' = 'http://schema-registry:8081',
    'auto.create'                         = 'true',
    'pk.mode'                             = 'record_key',
    'pk.fields'                           = 'request_id',
    'insert.mode'                         = 'upsert',
    'delete.enabled'                      = 'true'
);
```

# START the current best IPIG neural network model

This section will help to get data and predict events in real time

Please make sure you have all required data files: validation data, pipeline data, model checkpoint

`python .\push_data.py -b localhost:9092 -s http://127.0.0.1:8081`

