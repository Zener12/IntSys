import platform
import time
import json
import yaml
import argparse
import hashlib
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone

import csv
import uuid

import random
import numpy as np

import torch
import torch.nn as nn
import sklearn
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import optuna
import mlflow


class FirstNN(nn.Module):
    def __init__(self, hidden, n_layers, dropout):
        super().__init__()
        layers = []
        in_features = 64

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden

        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def make_loaders(batch_size, generator):
    root = Path(__file__).resolve().parents[2]
    split_path = root / "data" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))

    features, labels = load_digits(return_X_y=True)
    features = torch.tensor(features, dtype=torch.float32) / 16.0
    labels = torch.tensor(labels, dtype=torch.long)

    train_indices = torch.tensor(split["train"], dtype=torch.long)
    val_indices = torch.tensor(split["val"], dtype=torch.long)
    test_indices = torch.tensor(split["test"], dtype=torch.long)

    train_dataset = TensorDataset(features[train_indices], labels[train_indices], train_indices)
    val_dataset = TensorDataset(features[val_indices], labels[val_indices], val_indices)
    test_dataset = TensorDataset(features[test_indices], labels[test_indices], test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_nn(
        device, model, optimizer, 
        criterion, train_loader, 
        val_loader, max_epochs, 
        on_epoch_end=None):
    history = []

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for X_batch, y_batch, _ in tqdm(train_loader, desc=f"Train {epoch + 1}"):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_examples = 0

        with torch.no_grad():
            for X_val_batch, y_val_batch, _ in tqdm(val_loader, desc=f"Validation {epoch + 1}"):
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                output = model(X_val_batch)
                loss = criterion(output, y_val_batch)

                batch_size = y_val_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (output.argmax(dim=1) == y_val_batch).sum().item()
                val_examples += batch_size

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / train_examples,
            "val_loss": val_loss_sum / val_examples,
            "val_acc": val_correct / val_examples,
            "epoch_time": epoch_seconds
        }
        history.append(epoch_record)

        if on_epoch_end is not None:
            on_epoch_end(epoch_record)

    return history

def val_predict(model, loader, device):
    prediction_rows = []
    model.eval()
    
    for X_val_batch, y_val_batch, val_indices in tqdm(loader):
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        
        with torch.no_grad():
            output = model(X_val_batch)
            predictions = output.argmax(dim=1)

            indices = val_indices.cpu().tolist()
            true_labels = y_val_batch.cpu().tolist()
            predicted_labels = predictions.cpu().tolist()

            for index, y_true, y_pred in zip(indices, true_labels, predicted_labels):
                prediction_rows.append({
                    "split": "val",
                    "index": index,
                    "y_true": y_true,
                    "y_pred": y_pred
                })
    
    return prediction_rows

def test_accuracy(model, loader, device):
    prediction_rows = []
    model.eval()
    test_correct = 0
    test_examples = 0
    for X_test_batch, y_test_batch, test_indices in tqdm(loader, desc="Test"):
        with torch.no_grad():
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            output = model(X_test_batch)
            predictions = output.argmax(dim=1)

            batch_size = y_test_batch.size(0)
            test_correct += (output.argmax(dim=1) == y_test_batch).sum().item()
            test_examples += batch_size

            indices = test_indices.cpu().tolist()
            true_labels = y_test_batch.cpu().tolist()
            predicted_labels = predictions.cpu().tolist()
        
            for index, y_true, y_pred in zip(indices, true_labels, predicted_labels):
                prediction_rows.append({
                    "split": "test",
                    "index": index,
                    "y_true": y_true,
                    "y_pred": y_pred
                })

    accuracy = test_correct / test_examples

    return accuracy, prediction_rows


if __name__ == "__main__":
    metadata_time_start = datetime.now(timezone.utc).isoformat()

    root = Path(__file__).resolve().parents[2]
    
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip() 

    if status:
        raise SystemExit("Сначала проведите коммит изменений!")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--series", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["train"]["lr"] <= 0:
        raise SystemExit(f"train.lr должен быть больше 0; получено: {config['train']['lr']}")

    seed = config["seed"] if args.seed is None else args.seed
    
    run_id = uuid.uuid4().hex[:12]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # ==== START metadata.json (1) ====

    split_path = root / "data" / "split.json"
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    metadata = {
        "run_id": run_id,
        "series": args.series,
        "num_threads": config["runtime"]["num_threads"],
        "status": "running",
        "start_time": metadata_time_start,
        "end_time": None,
        "commit": commit,
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Scikit-learn version": sklearn.__version__,
        "Optuna version": optuna.__version__,
        "MLFlow version": mlflow.__version__,
        "seed": seed,
        "SHA-256": split_sha256
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8"
    )

    # ==== END metadata.json (1) ====

    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("lr0")
        mlflow_run = mlflow.start_run(run_name=run_id)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()
        metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8"
        )
        raise

    with mlflow_run:
        try:
            # ==== START config.yaml ====

            config["seed"] = seed
            config_path_in_run = run_dir / "config.yaml"
            config_path_in_run.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8"
            )
            print("Записана config.yaml")

            # ==== END config.yaml ====

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            device = config["runtime"]["device"]
            torch.set_num_threads(config["runtime"]["num_threads"])

            generator = torch.Generator()
            generator.manual_seed(seed)
            train_loader, val_loader, test_loader = make_loaders(
                config["train"]["batch_size"], generator
            )
            model = FirstNN(
                hidden=config["model"]["hidden"],
                n_layers=config["model"]["n_layers"],
                dropout=config["model"]["dropout"]
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=config["train"]["lr"], 
                                        weight_decay=config["train"]["weight_decay"]
            )

            mlflow.log_params({
                "seed": seed,
                "split_seed": config["split_seed"],
                "model.n_layers": config["model"]["n_layers"],
                "model.hidden": config["model"]["hidden"],
                "model.dropout": config["model"]["dropout"],
                "train.lr": config["train"]["lr"],
                "train.weight_decay": config["train"]["weight_decay"],
                "train.batch_size": config["train"]["batch_size"],
                "train.epochs": config["train"]["epochs"],
                "runtime.device": config["runtime"]["device"],
                "runtime.num_threads": config["runtime"]["num_threads"],
                "eval_test": args.eval_test
            })

            mlflow.set_tags({
                "commit": commit,
                "series": args.series,
                "seed": str(seed),
                "harness_run_id": run_id
            })

            # ==== START history.csv & predictions.csv ====

            history_path = run_dir / "history.csv"
            with history_path.open("w", newline="", encoding="utf-8") as file:
                def save_epoch(row):
                    writer.writerow(row)
                    file.flush()
                    for name in ("train_loss", "val_loss", "val_acc", "epoch_time"):
                        mlflow.log_metric(
                            name,
                            float(row[name]),
                            step=int(row["epoch"])
                        )

                writer = csv.DictWriter(
                    file,
                    fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "epoch_time"]
                )
                writer.writeheader()
                history = train_nn(
                            device, model, optimizer, criterion, train_loader, val_loader, 
                            max_epochs=config["train"]["epochs"], on_epoch_end=save_epoch
                )

            print("Записан history.csv")

            val_rows = val_predict(model, val_loader, device)

            test_acc = None
            test_rows = []
            if args.eval_test:
                test_acc, test_rows = test_accuracy(model, test_loader, device)

            predictions_path = run_dir / "predictions.csv"
            with predictions_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames = ["split", "index", "y_true", "y_pred"]
                )
                writer.writeheader()
                writer.writerows(val_rows)
                if args.eval_test:
                    writer.writerows(test_rows)

            print("Записан predictions.csv")

            # ==== END history.csv & predictions.csv ====

            # ==== START metrics.json ====

            with history_path.open("r", newline="", encoding="utf-8") as file:
                history_rows = list(csv.DictReader(file))

            with predictions_path.open("r", newline="", encoding="utf-8") as file:
                predictions_rows = list(csv.DictReader(file))

            val_rows = [row for row in predictions_rows if row["split"] == "val"]

            metrics = {
                "val_acc": sum(
                    int(row["y_true"]) == int(row["y_pred"]) for row in val_rows
                ) / len(val_rows),
                "val_loss": float(history_rows[-1]["val_loss"]),
                "train_time_seconds": sum(
                    float(row["epoch_time"]) for row in history_rows
                ),
            }

            if args.eval_test:
                test_rows = [row for row in predictions_rows if row["split"] == "test"]
                metrics["test_acc"] = sum(
                    int(row["y_true"]) == int(row["y_pred"]) for row in test_rows
                ) / len(test_rows)

            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8"
            )

            mlflow.log_metrics(metrics)
            print("Записан metrics.json")

            # ==== END metrics.json ====

            # ==== START metadata.json(2) ====

            metadata["status"] = "completed"
            metadata["end_time"] = datetime.now(timezone.utc).isoformat()
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8"
            )
            print("Записана metadata.json")
            
            mlflow.log_artifacts(str(run_dir))
 
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            metadata["traceback"] = traceback.format_exc()
            metadata["end_time"] = datetime.now(timezone.utc).isoformat()
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8"
            )
            print("Записана metadata.json")
            
            try:
                mlflow.log_artifacts(str(run_dir))
            except Exception as artifact_exc:
                metadata["artifact_logging_error"] = str(artifact_exc)
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n",
                    encoding="utf-8"
                )
            
            raise

            # ==== END metadata.json (2) ====
