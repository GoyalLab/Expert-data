from snakemake.io import load_configfile
import hashlib
import os
import yaml
import numpy as np
import pandas as pd
from src.statics import DATA_SHEET_KEYS, STRINGS, BOOLEANS, INTS, FLOATS
import logging


def recursive_update(d1, d2):
    for k, v in d2.items():
        if (
            k in d1 and isinstance(d1[k], dict) and isinstance(v, dict)
        ):
            recursive_update(d1[k], v)
        else:
            d1[k] = v

def load_configs(wf, default_config="config/defaults.yaml"):
    # Load default config
    config = load_configfile(default_config)
    
    # Update with any additional config files (from --configfile)
    for cf in wf.overwrite_configfiles:
        update_config = load_configfile(cf)
        recursive_update(config, update_config)

    # Update with command-line options (from --config)
    recursive_update(config, wf.overwrite_config)
    # Check config parameters
    check_config(config)
    return config

def hash(s: str, d: int = 4):
    return hashlib.shake_128(s.encode('utf-8')).hexdigest(d)

def get_param_hash(p: dict, dataset_indices: list[str], d: int = 8):
    blacklist = {'log_dir', 'output_dir', 'cache_dir', 'plot_dir', 'resources',
                  'benchmark_sizes', 'benchmark_n_folds', 'benchmark_context_groups',
                  # Storage-only knob: changing it must not relocate OUTPUT_DIR or
                  # invalidate cached results (same end data, different compression).
                  'defer_compression'}
    ps = p.copy()
    # sort dict
    ps = dict(sorted(p.items()))
    s = ';'.join([f'{k}:{v}' for k,v in ps.items() if k not in blacklist])
    # add all dataset indices
    s += 'datasets:' + ','.join(sorted(dataset_indices))
    return hash(s, d)

# Parameters that affect a SINGLE dataset's canonical block and sidecars.  Everything
# else -- the dataset list, the perturbation pool, min_cells_per_perturbation, merge /
# split / benchmark settings -- is applied at selection time and must NOT invalidate a
# stored block.  Splitting the hash this way is what makes adding dataset N+1 cheap:
# with one global hash, appending a dataset moved OUTPUT_DIR and orphaned every
# per-dataset artifact even though none of their inputs had changed.
DATASET_PARAM_KEYS = (
    'qc',
    'n_ctrl',
    'seed',
    'single_perturbations_only',
    'perturbation_col',
    'ctrl_key',
    # n_hvg is included because canonicalize stores the `highly_variable` /
    # `highly_variable_rank` columns using it.  The underlying `variances_norm` is
    # n_hvg-independent, so this could be relaxed later by having selection re-rank.
    'n_hvg',
)


def get_dataset_hash(p: dict, vocab_version: str, d: int = 8) -> str:
    """Hash of the params that determine one dataset's canonical block.

    Deliberately excludes the dataset list -- a block is a pure function of its raw
    file, these parameters, and the gene vocabulary.
    """
    items = {k: p.get(k) for k in DATASET_PARAM_KEYS}
    items['vocab_version'] = vocab_version
    s = ';'.join(f'{k}:{items[k]}' for k in sorted(items))
    return hash(s, d)


# Parameters that determine the MATERIALISED DB, on top of the dataset list.  An
# explicit allow-list, not a blacklist: with a blacklist any unrelated new config key
# silently invalidates every existing DB and forces a ~6 h rebuild.  Membership here is
# a deliberate choice.
RECIPE_PARAM_KEYS = (
    # gene axis
    'n_hvg',
    'var_merge',
    # perturbation pool
    'min_cells_per_class',
    'min_dataset_frac',
    'min_cells_per_perturbation',
    # zero-shot output
    'zs_min_cells_per_perturbation',
    'zs_eval_cap',
    # labels
    'perturbation_col',
    'ctrl_key',
    'seed',
)


def get_recipe_hash(p: dict, dataset_indices: list[str], d: int = 8) -> str:
    """Hash of the selection parameters + dataset list: keys the materialised DB.

    Deliberately NOT get_param_hash: that hashes every non-blacklisted key, so adding
    an unrelated setting to defaults.yaml would orphan the DB.
    """
    items = {k: p.get(k) for k in RECIPE_PARAM_KEYS}
    s = ';'.join(f'{k}:{items[k]}' for k in sorted(items))
    s += 'datasets:' + ','.join(sorted(dataset_indices))
    return hash(s, d)


def save_config(c: dict, o: str):
    d = os.path.dirname(o)
    os.makedirs(d, exist_ok=True)
    with open(o, 'w') as f:
        yaml.dump(c, f, default_flow_style=False)

def get_default_resources(resource_config: dict, default_key: str = 'default') -> dict[str, str]:
    o = {
        'time': resource_config['time'][default_key],
        'mem': resource_config['memory'][default_key],
    }
    partition = resource_config['partitions'][default_key]
    if partition != '':
        o['partition'] = partition
    return o

# With optional job name to lookup local configs
def estimate_resources(file_p: str, resource_config: dict, factor: int = 2, job_name: str | None = None, step: int = 10) -> str:
    # Get default memory options
    max_mem = resource_config['memory']['max_mem']
    min_mem = resource_config['memory']['min_mem']
    # Get jobs-specific resources
    job_mem_str = resource_config.get('jobs', {}).get(job_name, {}).get('mem')
    if job_mem_str is not None:
        # Set minimum memory to job-specific memory if given
        job_mem_str = job_mem_str.lower().rstrip('gb')
        min_mem = int(job_mem_str)
    # Estimate usage from file size
    bytes = os.path.getsize(file_p) * factor
    # Convert to GB
    gb_mem = round(bytes / 1024 ** 3)
        
    # Get partitions
    default_partition = resource_config['partitions']['default']
    highmem_partition = resource_config['partitions']['high_mem']
    # Fall back to minimum memory if its below that
    gb_mem = min_mem if gb_mem < min_mem else gb_mem
    # Set ram in steps
    if step > 0:
        gb_mem = int(step - gb_mem % step + gb_mem)
    # Assign partition
    partition = default_partition if gb_mem < max_mem else highmem_partition
    # Add gb string
    mem_str = f'{gb_mem}GB'
    # Return estimated resources
    return {
        'mem': mem_str,
        'partition': partition
    }

def get_job_resources(resource_config: dict, job_name: str, partition_prio: str | None = None, **kwargs) -> dict[str, str]:
    # Check if there are resource specification for that job name
    if job_name not in resource_config['jobs']:
        # NB: this used to call get_default_resources() with no arguments, which raises
        # TypeError instead of falling back -- so a rule with no resources entry failed
        # at DAG construction with an opaque error rather than using the defaults.
        logging.warning(
            f'No resource config for job {job_name!r}; falling back to defaults.')
        return get_default_resources(resource_config)
    # Get resource configs for job from defaults.yaml with updates by user
    time: str = resource_config['jobs'][job_name]['time']
    mem: str = resource_config['jobs'][job_name]['mem']
    threads: int = resource_config['jobs'][job_name].get('threads')
    partition: str = resource_config['partitions']['default']
    # Get maximum memory cap of default partition
    max_default_mem: int = resource_config['memory']['max_mem']
    # Try to set to high memory partition if given
    if int(mem.rstrip('GB')) > max_default_mem:
        hi_mem_partition = resource_config['partitions']['high_mem']
        if hi_mem_partition != '':
            partition = hi_mem_partition
        # Else set memory to default partition limit
        else:
            mem = str(max_default_mem)+'GB'
    # Try to set partition to given prio if found in config, fall back to previous partition if not
    if partition_prio is not None:
        partition = resource_config['partitions'].get(partition_prio, partition)
    o = {'time': time, 'mem': mem}
    if partition != '':
        o['partition'] = partition
    if threads is not None:
        o['threads'] = threads
    # Add any extra options to resources
    ext_kwargs = {k: v for k, v in resource_config['jobs'][job_name].items() if k not in o.keys()}
    o.update(ext_kwargs)
    # Add manual extra options to resources
    o.update(kwargs)
    return o


def determine_resources(config: dict, data_sheet: pd.DataFrame):
    # Update configs resource requirements based on the size of the input file
    resource_config: dict = config['resources']
    # Determine overall memory range
    mem_config: dict = resource_config['memory']
    min_mem = mem_config['min_mem']
    max_mem = mem_config['max_mem']
    # Determine partitions
    partition_config: dict = resource_config['partitions']
    default_partition: str = partition_config['default']
    gpu_partition: str = partition_config.get('gpu', None)
    # Check if there is a high memory allocation
    himem_alloc: str = partition_config.get('high_mem', default_partition)
    # Only overwrite configs if memory is not given
    if DATA_SHEET_KEYS.MEM not in data_sheet.columns:
        # Fill memory column based on bytes given in metadata
        if DATA_SHEET_KEYS.BYTES in data_sheet.columns:
            mem_per_ds = np.maximum((data_sheet[DATA_SHEET_KEYS.BYTES] / data_sheet[DATA_SHEET_KEYS.BYTES].max() * max_mem).astype(int), min_mem)
            data_sheet[DATA_SHEET_KEYS.MEM] = mem_per_ds.astype(str) + 'GB'
        else:
            # Fall back to minimum memory settings
            data_sheet[DATA_SHEET_KEYS.MEM] = min_mem
    else:
        # Check if memory is given in correct format
        assert data_sheet[DATA_SHEET_KEYS.MEM].str.endswith('GB').all()
    # Set max memory to 2x MEM or max_mem if that is less
    def calc_max_mem(mem_str):
        mem_val = int(mem_str.rstrip('GB'))
        doubled = 2 * mem_val
        return f"{min(doubled, max_mem)}GB"
    data_sheet[DATA_SHEET_KEYS.MAX_MEM] = data_sheet[DATA_SHEET_KEYS.MEM].apply(calc_max_mem)
    # Set partition for each dataset based on memory assigned to it
    data_sheet[DATA_SHEET_KEYS.PARTITION] = default_partition
    himem_mask = data_sheet[DATA_SHEET_KEYS.MEM].str.rstrip('GB').astype(int) > max_mem
    highest_mem = data_sheet[DATA_SHEET_KEYS.MEM].str.rstrip('GB').astype(int).max()
    if himem_mask.sum() > 0 and himem_alloc == default_partition:
        logging.warning(f'Some datasets allocate more memory than the set max. memory ({max_mem}) and no high-memory partition is given. Defaulting to partition {default_partition} and max. allocation of {highest_mem}GB.')
    data_sheet.loc[himem_mask,DATA_SHEET_KEYS.PARTITION] = himem_alloc

def get_perturbation_col_override(data_sheet: pd.DataFrame, dataset: str) -> str | None:
    """Return the per-dataset source perturbation-column override, or None.

    Looks up the optional ``perturbation_col`` datasheet column for ``dataset``;
    returns its value when set (non-empty / non-NaN), otherwise None so callers
    fall back to P_COLS auto-detection.
    """
    col = DATA_SHEET_KEYS.PERTURBATION_COL
    if col not in data_sheet.columns or dataset not in data_sheet.index:
        return None
    val = data_sheet.loc[dataset, col]
    if pd.isna(val) or str(val).strip() == '':
        return None
    return str(val).strip()


def read_data_sheet(config: dict):
    d = pd.read_csv(str(config['datasheet']))
    if DATA_SHEET_KEYS.INDEX not in d.columns:
        d[DATA_SHEET_KEYS.INDEX] = d[DATA_SHEET_KEYS.P_INDEX].astype(str) + d[DATA_SHEET_KEYS.D_INDEX].fillna('').apply(lambda x: f"_{x}" if x else '')
    # set as index
    d.set_index(DATA_SHEET_KEYS.INDEX, inplace=True)
    if DATA_SHEET_KEYS.CANCER in d.columns:
        # convert y/n to boolean
        d[DATA_SHEET_KEYS.CANCER] = d[DATA_SHEET_KEYS.CANCER].map({'y': True, 'n': False})
    # determine resources for each dataset
    determine_resources(config, d)
    return d


def correction_methods():
    return ['scANVI', 'scanorama', 'harmonypy', 'skip']

def requires_raw_data():
    return ['scANVI']

def requires_processed_data():
    return ['scanorama', 'harmonypy']

def requires_gpu():
    return ['scANVI']

def check_method(conf):
    import warnings

    methods = correction_methods()
    m = conf['correction_method']
    if m not in methods:
        raise ValueError(f'Method must be {methods}; got {m}')
    if m in requires_raw_data():
        logging.info(f'Method {m} requires raw data, setting preprocess to exclude normalization and log1p')
        conf['norm'] = False
        conf['log_norm'] = False
        conf['scale'] = False
    if m in requires_processed_data():
        logging.info(f'Method {m} requires preprocessed data, setting preprocess to include normalization and log1p')
        conf['norm'] = True
        conf['log_norm'] = True
    if m in requires_gpu():
        import torch
        if not torch.cuda.is_available():
            # raise SystemError('scANVI requires a GPU to effectively harmonize.')
            warnings.warn('scANVI requires a GPU to effectively harmonize. Running on CPU will significantly increase the runtime.')

def check_types(conf):
    type_checks = {
        str: STRINGS,
        bool: BOOLEANS,
        int: INTS,
        float: FLOATS
    }
    # Check every param for correct type
    for param, value in conf.items():
        for expected_type, keys in type_checks.items():
            if param in keys:
                assert isinstance(value, expected_type), f"Invalid value for {expected_type.__name__} parameter '{param}': {value}"

def check_config(conf: dict):
    # validate config types
    check_types(conf)
    # Harmonisation is a legacy-only stage; the canonical-store pipeline has no
    # correction_method, so only validate it when the key is actually present.
    if 'correction_method' in conf:
        check_method(conf)
    # add individual resource requirements
