from src.utils import setup_logger
from src.merge import merge


def merge_datasets(
        dataset_files: list[str],
        merged_set_output: str | None = None,
        merge_method: str = 'concat_on_disk',
        ctrl_pth: str | None = None,
        pert_pth: str | None = None,
        perturbation_col: str = 'perturbation',
        ctrl_key: str = 'control',
        compression: str | None = 'gzip',
    ) -> None:
    dataset_files = sorted(dataset_files)
    merge(dataset_files, out_pth=merged_set_output, method=merge_method,
          ctrl_pth=ctrl_pth, pert_pth=pert_pth,
          perturbation_col=perturbation_col, ctrl_key=ctrl_key,
          compression=compression)


if __name__ == "__main__":
    log_file = snakemake.log[0]
    setup_logger(log_file)
    try:
        merge_datasets(
            dataset_files=snakemake.input.dataset_files,
            merged_set_output=getattr(snakemake.output, 'merged_set', None),
            merge_method=snakemake.params.merge_method,
            ctrl_pth=snakemake.params.get('ctrl_pth', None),
            pert_pth=snakemake.params.get('pert_pth', None),
            perturbation_col=snakemake.params.get('perturbation_col', 'perturbation'),
            ctrl_key=snakemake.params.get('ctrl_key', 'control'),
            compression=snakemake.params.get('compression', 'gzip'),
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
