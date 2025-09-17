#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from loguru import logger


def migrate_data(src_root, dst_root):
    """
    Migrate all data from src_root to dst_root.
    Skip copying if the destination already exists and has the same size, otherwise copy.
    Files with different sizes will be updated.

    Args:
        src_root (str): Source directory path
        dst_root (str): Destination directory path
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    if not src_path.exists():
        logger.error(f"Source directory '{src_root}' does not exist")
        return False

    if not src_path.is_dir():
        logger.error(f"Source path '{src_root}' is not a directory")
        return False

    # Ensure destination root directory exists
    dst_path.mkdir(parents=True, exist_ok=True)

    migrated_count = 0
    updated_count = 0
    skipped_count = 0

    logger.info(f"Starting data migration from '{src_root}' to '{dst_root}'")

    try:
        # Iterate through all files and directories in source directory
        for src_file_path in src_path.rglob("*"):
            # Calculate relative path
            relative_path = src_file_path.relative_to(src_path)
            dst_file_path = dst_path / relative_path

            if '.venv' in src_file_path.parts:
                logger.debug(f"Skipping .venv: {relative_path}")
                continue

            if src_file_path.is_file():
                # If destination file doesn't exist, copy the file
                if not dst_file_path.exists():
                    # Ensure parent directory of destination file exists
                    dst_file_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Copying new file: {relative_path}")
                    shutil.copy2(src_file_path, dst_file_path)
                    migrated_count += 1
                else:
                    # Check file sizes to determine if we need to copy
                    src_size = src_file_path.stat().st_size
                    dst_size = dst_file_path.stat().st_size

                    if src_size != dst_size:
                        # Ensure parent directory of destination file exists
                        dst_file_path.parent.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Updating file (size mismatch {dst_size} -> {src_size}): {relative_path}")
                        shutil.copy2(src_file_path, dst_file_path)
                        updated_count += 1
                    else:
                        logger.debug(f"Skipping identical file (size {src_size}): {relative_path}")
                        skipped_count += 1

            elif src_file_path.is_dir():
                # For directories, create if doesn't exist
                if not dst_file_path.exists():
                    logger.debug(f"Creating directory: {relative_path}")
                    dst_file_path.mkdir(parents=True, exist_ok=True)
                    migrated_count += 1
                else:
                    logger.debug(f"Skipping existing directory: {relative_path}")
                    skipped_count += 1

        logger.info("Migration completed!")
        logger.info(f"New items: {migrated_count}")
        logger.info(f"Updated items: {updated_count}")
        logger.info(f"Skipped items: {skipped_count}")
        logger.info(f"Total processed: {migrated_count + updated_count + skipped_count}")
        return True

    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False


def setup_logger(log_level: str = "INFO", log_file: str = None):
    """Setup loguru logger with specified level and optional file output"""
    # Remove default handler
    logger.remove()

    # Set log level
    level = log_level.upper()
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        level = "INFO"

    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=level,
            rotation="10 MB",
            retention="1 week"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data from source directory to destination directory"
    )
    parser.add_argument(
        "src_root",
        help="Source directory path"
    )
    parser.add_argument(
        "dst_root",
        help="Destination directory path"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Set logging level (default: DEBUG)"
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path for saving logs",
        default="migrate.log"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(args.log_level, args.log_file)

    logger.info("Data migration script started")
    logger.info(f"Source: {args.src_root}")
    logger.info(f"Destination: {args.dst_root}")
    logger.info(f"Log level: {args.log_level}")

    success = migrate_data(args.src_root, args.dst_root)

    if success:
        logger.info("Data migration script completed successfully")
    else:
        logger.error("Data migration script failed")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
