"""The pipeline worker: one segment.ready in, detection files and Unity requests out (no broker, no GPU)."""
import json
from pathlib import Path

import pytest
import yaml

from triffid_uav.worker import WorkerSettings, process_segment


@pytest.fixture
def worker(project, tmp_path):
    """Settings like the pipeline's: the segment and its SRT named after the dataset, outputs per dataset."""
    segments, telemetry = tmp_path / 'segments', tmp_path / 'telemetry'
    segments.mkdir()
    telemetry.mkdir()
    (project / 'inputs' / 'flight.avi').rename(segments / 'flight_001.avi')
    (project / 'inputs' / 'flight.srt').rename(telemetry / 'flight_001.srt')
    config = tmp_path / 'worker.yaml'
    config.write_text(yaml.safe_dump({'paths': {'supports': str(project / 'supports')}}))
    settings = WorkerSettings(telemetry_dir=telemetry, output_dir=tmp_path / 'perception', config_path=config)
    published = []
    return settings, segments / 'flight_001.avi', published, lambda key, message: published.append((key, message))


def test_flight_detections_become_unity_requests_with_the_flight_id(worker, fake_backend):
    settings, segment, published, publish = worker
    payload = {'job_id': 'upload-42', 'segment_path': str(segment)}
    requests = process_segment(payload, settings, fake_backend, publish)

    assert [key for key, _ in published] == ['unity.ready', 'unity.ready']
    points, polygons = (message for _, message in published)
    assert requests == [points, polygons]
    outputs = settings.output_dir / 'flight_001'

    assert points['action'] == 'project_points'
    assert points['flight_id'] == polygons['flight_id'] == 'upload-42'
    assert points['dataset_id'] == 'flight_001'
    assert Path(points['observations_json_path']) == outputs / 'point_detections.json'
    document = json.loads(Path(points['observations_json_path']).read_text())
    assert document['schema_version'] == 'triffid.uav.observations.v1'
    # Unity matches each view to the staged flight's SRT through this telemetry.
    assert all(s['telemetry']['frame_index'] >= 0 and 'timestamp_s' in s['telemetry'] for s in document['sources'])

    assert polygons['action'] == 'project_masks'
    assert polygons['photo_srt_path'] == str(settings.telemetry_dir / 'flight_001.srt')
    polygon_document = json.loads(Path(polygons['observations_json_path']).read_text())
    source = polygon_document['sources'][0]
    # 3c: the most downward frame of the flight SRT itself; every block is at -45, so the earliest.
    assert polygons['reference_frame'] == source['frame_count'] == 1
    assert source['source_type'] == 'video_frame'


def test_a_reupload_replaces_the_earlier_outputs(worker, fake_backend):
    settings, segment, _, publish = worker
    stale = settings.output_dir / 'flight_001' / 'stale.txt'
    stale.parent.mkdir(parents=True)
    stale.write_text('previous upload')
    process_segment({'job_id': 'upload-43', 'segment_path': str(segment)}, settings, fake_backend, publish)
    assert not stale.exists()
    assert (settings.output_dir / 'flight_001' / 'point_detections.json').is_file()


def test_nothing_detected_sends_no_request(worker, fake_backend):
    settings, segment, published, publish = worker
    fake_backend.predict = lambda rgb, classes: [
        type('P', (), {'class_id': c.id, 'mask': (rgb[..., 0] < 0), 'probabilities': None})() for c in classes]
    assert process_segment({'job_id': 'upload-44', 'segment_path': str(segment)}, settings, fake_backend, publish) == []
    assert published == []


def test_missing_srt_fails_before_any_work(worker, fake_backend):
    settings, segment, published, publish = worker
    (settings.telemetry_dir / 'flight_001.srt').unlink()
    with pytest.raises(FileNotFoundError, match='flight_001.srt'):
        process_segment({'job_id': 'upload-45', 'segment_path': str(segment)}, settings, fake_backend, publish)
    assert published == [] and fake_backend.prepared == 0
