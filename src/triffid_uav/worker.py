"""RabbitMQ worker: detects on every uploaded flight and asks Unity to project the detections.

    segment.ready (ingest; the same event gsplat consumes, on a queue of our own)
      -> run() on <segment>.mp4 + <TELEMETRY_DIR>/<dataset>.srt, the SRT ingest wrote
      -> <PERCEPTION_OUTPUT_DIR>/<dataset>/point_detections.json, polygon_detections.json, run.json
      -> unity.ready: project_points and project_masks, each with flight_id = the segment.ready job_id

Perception and gsplat run in parallel. Unity holds a request whose flight it has not staged yet
in a waiting queue and runs it once gsplat's stage job for that flight is done, so neither side
waits for the other here.

Polygons run on the frame of the flight SRT whose gimbal pitch is closest to -90 degrees (the
earliest one on a tie). The flights so far have no nadir frame, so this is an oblique frame;
see the README, "Polygons in the pipeline".

Paths in the requests are the paths this container sees; the Unity container mounts the same
data folder at the same place (/data in the pipeline's docker-compose.yml).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
import shutil
import time

from .config import load_config
from .models import build_backend
from .pipeline import run

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSettings:
    rabbitmq_host: str = 'rabbitmq'
    rabbitmq_port: int = 5672
    rabbitmq_user: str = 'guest'
    rabbitmq_pass: str = 'guest'
    exchange: str = 'pipeline.events'
    queue: str = 'perception.queue'
    segment_routing_key: str = 'segment.ready'
    unity_routing_key: str = 'unity.ready'
    error_routing_key: str = 'pipeline.error'
    # Inference blocks the connection for minutes; heartbeats would drop it.
    heartbeat: int = 0
    telemetry_dir: Path = Path('/data/telemetry')
    output_dir: Path = Path('/data/perception')
    config_path: Path = Path('configs/default.yaml')

    @classmethod
    def from_env(cls) -> 'WorkerSettings':
        env = os.environ.get
        return cls(
            rabbitmq_host=env('RABBITMQ_HOST', cls.rabbitmq_host),
            rabbitmq_port=int(env('RABBITMQ_PORT', cls.rabbitmq_port)),
            rabbitmq_user=env('RABBITMQ_USER', cls.rabbitmq_user),
            rabbitmq_pass=env('RABBITMQ_PASS', cls.rabbitmq_pass),
            exchange=env('RABBITMQ_EXCHANGE', cls.exchange),
            queue=env('RABBITMQ_PERCEPTION_QUEUE', cls.queue),
            segment_routing_key=env('RABBITMQ_SEGMENT_ROUTING_KEY', cls.segment_routing_key),
            unity_routing_key=env('RABBITMQ_UNITY_ROUTING_KEY', cls.unity_routing_key),
            error_routing_key=env('RABBITMQ_ERROR_ROUTING_KEY', cls.error_routing_key),
            heartbeat=int(env('RABBITMQ_HEARTBEAT', cls.heartbeat)),
            telemetry_dir=Path(env('TELEMETRY_DIR', str(cls.telemetry_dir))),
            output_dir=Path(env('PERCEPTION_OUTPUT_DIR', str(cls.output_dir))),
            config_path=Path(env('PERCEPTION_CONFIG', str(cls.config_path))),
        )


def flight_config(settings: WorkerSettings, segment: Path, srt: Path, outputs: Path):
    """The run configuration for one flight: the configured model and classes, this flight's files."""
    overrides = [
        f'paths.outputs={outputs}',
        f'paths.flight_video={segment}',
        f'paths.flight_srt={srt}',
        # 3c: polygons on the most downward frame of the flight itself.
        f'paths.topview_srt={srt}',
        # No nadir photo: the frame comes from the video. The topview folder is the fresh output
        # folder, so no stray image can be picked up as one.
        f'paths.topview={outputs}',
        'paths.topview_image=null',
    ]
    return load_config(settings.config_path, overrides)


def process_segment(payload: dict, settings: WorkerSettings, backend, publish) -> list[dict]:
    """Detect on one flight and publish its Unity requests. Returns the requests published.

    `publish(routing_key, message)` sends one message; `backend` is loaded once by the caller.
    """
    flight_id = payload['job_id']
    segment = Path(payload['segment_path'])
    dataset = segment.stem
    srt = settings.telemetry_dir / f'{dataset}.srt'
    if not segment.is_file():
        raise FileNotFoundError(f'Segment does not exist: {segment}')
    if not srt.is_file():
        raise FileNotFoundError(f'Flight SRT not found: {srt} (ingest writes it next to the segment)')

    outputs = settings.output_dir / dataset
    if outputs.exists():
        log.warning('Replacing the earlier perception outputs of dataset %s: %s', dataset, outputs)
        shutil.rmtree(outputs)
    outputs.mkdir(parents=True)

    log.info('Detecting on dataset %s (flight_id %s): %s + %s', dataset, flight_id, segment, srt)
    summary = run(flight_config(settings, segment, srt, outputs), backend=backend)

    common = {'flight_id': flight_id, 'dataset_id': dataset}
    requests = []
    points = sum(summary['counts']['points_per_frame'].values())
    if 'points' in summary['outputs']:
        if points:
            requests.append({'job_id': f'{flight_id}-points', 'action': 'project_points',
                             'observations_json_path': summary['outputs']['points'], **common})
        else:
            log.info('Dataset %s: no point detections in %d orbit frames; nothing to project',
                     dataset, len(summary['counts']['points_per_frame']))
    polygons = summary['counts']['polygon_components']
    if 'polygons' in summary['outputs']:
        if polygons:
            requests.append({'job_id': f'{flight_id}-polygons', 'action': 'project_masks',
                             'observations_json_path': summary['outputs']['polygons'],
                             'photo_srt_path': str(srt), 'reference_frame': summary['topview_frame_count'],
                             **common})
        else:
            log.info('Dataset %s: no polygon detections on FrameCnt %s; nothing to project',
                     dataset, summary['topview_frame_count'])

    for request in requests:
        publish(settings.unity_routing_key, request)
        log.info('Published %s for dataset %s (job_id %s)', request['action'], dataset, request['job_id'])
    log.info('Dataset %s done: %d point and %d polygon detections in %.1fs', dataset, points, polygons,
             summary['timings']['total_s'])
    return requests


def serve(settings: WorkerSettings | None = None, backend=None) -> None:
    """Consume segment.ready forever. The model is loaded once, before the first message."""
    import pika

    settings = settings or WorkerSettings.from_env()
    if backend is None:
        # Fails at start (missing weights, unknown backend) instead of on the first flight.
        backend = build_backend(load_config(settings.config_path).model)

    def connect():
        credentials = pika.PlainCredentials(settings.rabbitmq_user, settings.rabbitmq_pass)
        parameters = pika.ConnectionParameters(host=settings.rabbitmq_host, port=settings.rabbitmq_port,
                                               credentials=credentials, heartbeat=settings.heartbeat)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.exchange_declare(exchange=settings.exchange, exchange_type='topic', durable=True)
        channel.queue_declare(queue=settings.queue, durable=True)
        channel.queue_bind(queue=settings.queue, exchange=settings.exchange,
                           routing_key=settings.segment_routing_key)
        channel.basic_qos(prefetch_count=1)
        return connection, channel

    def on_message(channel, method, _properties, body):
        def publish(routing_key, message):
            channel.basic_publish(exchange=settings.exchange, routing_key=routing_key,
                                  body=json.dumps(message).encode(),
                                  properties=pika.BasicProperties(content_type='application/json',
                                                                  delivery_mode=2))
        payload = None
        try:
            payload = json.loads(body)
            process_segment(payload, settings, backend, publish)
        except Exception as error:  # noqa: BLE001 - every failure is reported, then the next flight runs
            log.exception('Perception failed for %s', payload)
            publish(settings.error_routing_key, {
                'job_id': payload.get('job_id', 'unknown') if isinstance(payload, dict) else 'unknown',
                'error': str(error), 'source': 'perception-worker', 'timestamp': int(time.time())})
        channel.basic_ack(delivery_tag=method.delivery_tag)

    while True:
        connection = None
        try:
            connection, channel = connect()
            channel.basic_consume(queue=settings.queue, on_message_callback=on_message, auto_ack=False)
            log.info('Perception worker consuming %s (%s on %s); outputs in %s',
                     settings.queue, settings.segment_routing_key, settings.exchange, settings.output_dir)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as error:
            log.warning('RabbitMQ unavailable (%s); retrying in 2s', error)
            time.sleep(2)
        finally:
            if connection is not None and connection.is_open:
                connection.close()
