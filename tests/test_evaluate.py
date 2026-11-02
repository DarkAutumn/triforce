"""``triforce.evaluate``: per-specialist evaluation over every seed start state."""

from triforce.evaluate import evaluate_specialist
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import ObservationBuilder
from triforce.registry import save_checkpoint
from triforce.specialist import load_specialist


def test_evaluate_specialist_covers_every_seed_state_reproducibly(session, tmp_path):
    network = ZeldaPolicy(ObservationBuilder.observation_space(),
                          ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32))
    ckpt = save_checkpoint('boss-aquamentus', 'untrained', network, root=tmp_path)
    config = load_specialist('boss-aquamentus').model_copy(update={'max_steps': 40})

    result = evaluate_specialist(config, ckpt, episodes_per_state=2, session=session, seed=5)
    records = result['records']
    assert result['episodes'] == 6 == len(records)
    assert set(result['by_state']) == {'1_35s_0', '1_35s_1', '1_35s_2'}
    assert all(stats['episodes'] == 2 for stats in result['by_state'].values())
    assert sorted(r['seed'] for r in records) == list(range(5, 11))
    assert all(r['outcome'] in {'success', 'hard-failure', 'soft-failure'} for r in records)
    assert all(0 < r['steps'] <= 40 for r in records)
    assert result['success_rate'] == sum(r['success'] for r in records) / 6
    assert set(result['by_room']) == {'1:0x35'}

    again = evaluate_specialist(config, ckpt, episodes_per_state=2, session=session, seed=5)
    assert again['records'] == records
