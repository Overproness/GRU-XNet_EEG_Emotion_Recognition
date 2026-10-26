"""Check that repeated films, not clip counts, determine split feasibility."""
import importlib.util
from pathlib import Path
import sys

script = Path(__file__).resolve().parents[1] / 'scripts/analyze_faced_backup.py'
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('faced_analysis', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_more_clips_do_not_create_more_films():
    records = [{'clip': i, 'source_film_sha256': 'same-film', 'assigned_emotion': 'neutral'} for i in range(1, 5)]
    result = module.family_summary(records)
    assert result['exact_source_film_families'] == 1
    assert result['film_families_by_assigned_emotion'] == {'neutral': 1}
    assert result['repeated_film_clip_groups'] == [[1, 2, 3, 4]]
    assert not result['three_way_all_nine_class_film_disjoint_feasible']


def test_distinct_films_supply_necessary_three_way_support():
    records = [{'clip': i, 'source_film_sha256': str(i), 'assigned_emotion': 'fear'} for i in range(1, 4)]
    assert module.family_summary(records)['three_way_all_nine_class_film_disjoint_feasible']


def test_normalized_film_titles_keep_duplicate_sources_together():
    rows = [['index', 'seconds', 'film', 'db', 'valence', 'emotion']]
    rows += [[str(i), '30', 'FILM  A' if i == 1 else ' film a ' if i == 2 else str(i), 'db', 'negative', 'fear'] for i in range(1, 29)]
    records = module.stimulus_projection(rows)['records']
    assert records[0]['source_film_sha256'] == records[1]['source_film_sha256']
    assert records[2]['source_film_sha256'] != records[0]['source_film_sha256']
