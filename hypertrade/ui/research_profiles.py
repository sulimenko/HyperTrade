"""Reusable UI search configuration, separate from immutable experiment bundles."""
from dataclasses import asdict
import json
import math
from pathlib import Path
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE
from hypertrade.config.schemas import FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig

TRIAL_PRESETS = ['20 · smoke / research test', '100 · исследование', '200 · исследование', 'Вручную']


def trial_count(preset, custom=20):
    if preset not in TRIAL_PRESETS:
        raise ValueError('Неизвестный trial preset')
    value = custom if preset == 'Вручную' else int(preset.split(' ·')[0])
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10000:
        raise ValueError('n_trials должен быть целым числом от 1 до 10000')
    return value


def resolve_search(payload):
    if not isinstance(payload, dict):
        raise ValueError('Search JSON должен быть объектом')
    defaults = asdict(DEFAULT_FILTER_SEARCH_SPACE)
    for key, value in payload.items():
        if key not in defaults:
            raise ValueError(f'Неизвестное поле: {key}')
        expected = defaults[key]
        if isinstance(expected, bool):
            valid = type(value) is bool
        elif isinstance(expected, int):
            valid = type(value) is int
        elif isinstance(expected, float):
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
        if not valid:
            raise ValueError(f'Неверный тип/значение: {key}')
    result = FilterSearchSpace.from_dict(payload)
    for stem in ('atr_sl', 'atr_tp'):
        if getattr(result, stem+'_min') <= 0:
            raise ValueError(f'{stem}_min должен быть положительным')
    # Optuna otherwise silently shrinks a non-divisible high bound.
    for key, low in asdict(result).items():
        if key.endswith('_min') and hasattr(result, key[:-4]+'_step'):
            stem = key[:-4]
            ticks = (getattr(result, stem+'_max') - low) / getattr(result, stem+'_step')
            if not math.isclose(ticks, round(ticks), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'{stem}: (max − min) должен делиться на step без остатка')
    return result


def research_profile(objective=None, search=None, preset=TRIAL_PRESETS[0], n_trials=20, seed=42):
    profile = ObjectiveProfile.from_dict(objective or DEFAULT_OBJECTIVE_PROFILE.to_dict())
    resolved = resolve_search(search if search is not None else asdict(DEFAULT_FILTER_SEARCH_SPACE))
    count = trial_count(preset, n_trials)
    if type(n_trials) is not int or n_trials != count:
        raise ValueError('n_trials не соответствует выбранному trial preset')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError('Seed должен быть целым числом 0…4294967295')
    return dict(version=1, objective_profile=profile.to_dict(), search_space=asdict(resolved),
                trial_preset=preset, n_trials=count, seed=seed)


def load_profile(path):
    data = json.loads(Path(path).read_text())
    # Legacy objective-only profiles remain readable; saving migrates to runtime.
    if 'objective_profile' not in data:
        return research_profile(objective=data)
    return research_profile(data['objective_profile'], data['search_space'], data['trial_preset'], data['n_trials'], data['seed'])


def save_profile(root, config):
    validated = research_profile(config['objective_profile'], config['search_space'], config['trial_preset'], config['n_trials'], config['seed'])
    target = Path(root)/(validated['objective_profile']['name']+'.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(validated, ensure_ascii=False, indent=2))
    temporary.replace(target)
    return target


def objective_templates():
    default = research_profile()
    capital = research_profile(objective={**DEFAULT_OBJECTIVE_PROFILE.to_dict(), 'name': 'capital_return',
        'objectives': [dict(metric='portfolio_return_pct', direction='maximize')]})
    balanced = research_profile(objective={**DEFAULT_OBJECTIVE_PROFILE.to_dict(), 'name': 'portfolio_balance',
        'objectives': [dict(metric='portfolio_return_pct', direction='maximize'),
                       dict(metric='portfolio_max_drawdown_pct', direction='minimize'), dict(metric='profit_factor', direction='maximize')]})
    return {'Качество сигналов · текущий default': default, 'Доходность капитала · одна цель': capital,
            'Баланс доходности / просадки / PF': balanced}


def run_config(config, signals_path, label, artifacts_root):
    return OptimizationRunConfig(signals_path=signals_path, n_trials=trial_count(config['trial_preset'], config['n_trials']),
                                 seed=config['seed'], run_label=label, artifacts_root=str(artifacts_root))
