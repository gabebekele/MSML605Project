import os
import csv


def validate_pairs_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pairs file not found: {path}")

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Pairs file has no header row")

        required_columns = {"left_path", "right_path", "label", "split"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(f"Pairs file is missing required columns: {missing_columns}")

        valid_labels = {"0", "1"}
        valid_splits = {"train", "val", "test"}

        for row_number, row in enumerate(reader, start=2):
            if row["label"] not in valid_labels:
                raise ValueError(f"Invalid label at row {row_number}: {row['label']}")

            if row["split"] not in valid_splits:
                raise ValueError(f"Invalid split at row {row_number}: {row['split']}")

            if row["left_path"] == "" or row["right_path"] == "":
                raise ValueError(f"Missing image path at row {row_number}")


def validate_threshold(threshold):
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be numeric")

    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("Threshold must be between -1.0 and 1.0")


def validate_scores_and_labels(scores, labels):
    if len(scores) != len(labels):
        raise ValueError("The number of scores must match the number of labels")

    if len(scores) == 0:
        raise ValueError("Scores cannot be empty")

    for label in labels:
        if int(label) not in (0, 1):
            raise ValueError("Labels must be binary values")
