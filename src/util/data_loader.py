from src.util.logging_util import get_logger
import kagglehub
from pathlib import Path


def load_dataset(dataset_name: str):
    logger = get_logger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Find CSV files in the directory
        data_dir = Path(dataset_path)
        csv_files = list(data_dir.glob("*.csv"))

        if not csv_files:
            logger.error(f"No CSV files found in directory: {dataset_path}")
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        logger.info(
            f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}"
        )

        # Return the first CSV file path (or all if multiple)
        if len(csv_files) == 1:
            csv_file_path = str(csv_files[0])
            logger.info(f"Using CSV file: {csv_file_path}")
            return [csv_file_path]  # Return as list for consistency
        else:
            # Return list of all CSV files
            csv_file_paths = [str(f) for f in csv_files]
            logger.info(f"Multiple CSV files found, returning list of paths")
            return csv_file_paths

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
