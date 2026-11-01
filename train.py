from pathlib import Path
from typing import Literal

import numpy as np
from comet_ml import Experiment
import torch
import xarray as xr
from tap import Tap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import FNN, TransformerModel


class Parser(Tap):
    data_dir: Path = Path("data/preprocessed")
    model: Literal["fnn", "transformer"] = "transformer"
    epochs: int = 1
    batch_size: int = 1024
    learning_rate: float = 1e-3
    workers: int = 0
    seed: int = 0
    train_year: int = 2013
    test_year: int = 2014
    hidden_layers: list[int] = [128, 128]
    checkpoint_dir: Path = Path("checkpoints")
    resume: Path | None = None
    comet_project: str = "heating-rate-retrieval"


class RRTMGDataset(Dataset):
    def __init__(self, fluxes, heating_rates, flux_mean, flux_std, htr_mean, htr_std):
        self.fluxes = torch.from_numpy((fluxes - flux_mean) / flux_std)
        self.heating_rates = torch.from_numpy((heating_rates - htr_mean) / htr_std)

        print("invalid targets:", np.count_nonzero(~np.isfinite(heating_rates)))
        print("invalid means:", np.flatnonzero(~np.isfinite(htr_mean)))
        print("invalid stds:", np.flatnonzero(~np.isfinite(htr_std)))


    def __len__(self):
        return len(self.fluxes)

    def __getitem__(self, index):
        return self.fluxes[index], self.heating_rates[index]


def load_data(data_dir: Path, train_year: int, test_year: int):
    if train_year == test_year:
        raise ValueError("Training and testing years must differ")

    def load(year):
        fluxes = []
        heating_rates = []
        year_path = data_dir / f"data_{year}.nc"
        paths = (
            [year_path]
            if year_path.exists()
            else sorted(data_dir.glob(f"*_{year}.nc"))
        )
        if not paths:
            raise FileNotFoundError(f"No preprocessed data found for {year} in {data_dir}")
        for path in paths:
            with xr.open_dataset(path) as dataset:
                x = dataset.toa_flux_lw.transpose(
                    "time", "latitude", "longitude", "band"
                ).values.reshape(-1, dataset.sizes["band"])
                y = dataset.htr_lw.transpose(
                    "time", "latitude", "longitude", "level"
                ).values.reshape(-1, dataset.sizes["level"])
                fluxes.append(x.astype(np.float32))
                heating_rates.append(y.astype(np.float32))
        return np.concatenate(fluxes), np.concatenate(heating_rates)

    train_fluxes, train_htr = load(train_year)
    test_fluxes, test_htr = load(test_year)
    return train_fluxes, train_htr, test_fluxes, test_htr


def make_model(name: str, num_bands: int, num_levels: int, hidden_layers):
    if name == "fnn":
        return FNN(num_bands, [*hidden_layers, num_levels])
    return TransformerModel(num_bands=num_bands, num_levels=num_levels)


def save_checkpoint(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def masked_mse(predictions, targets):
    valid = torch.isfinite(targets)
    if not valid.any():
        raise ValueError("Batch contains no finite heating-rate targets")
    return torch.mean((predictions[valid] - targets[valid]) ** 2)


def main():
    args = Parser().parse_args()
    if not args.hidden_layers or any(size <= 0 for size in args.hidden_layers):
        raise ValueError("--hidden_layers must contain positive integers")
    experiment = Experiment(project_name=args.comet_project, parse_args=False)
    experiment.log_parameters(args.as_dict())

    torch.manual_seed(args.seed)
    train_fluxes, train_htr, test_fluxes, test_htr = load_data(
        args.data_dir, args.train_year, args.test_year
    )
    if not len(train_fluxes):
        raise ValueError("No training profiles found")
    # Axis 0 is the flattened (time, latitude, longitude) sample dimension.
    flux_mean = np.nanmean(train_fluxes, axis=0)
    flux_std = np.maximum(np.nanstd(train_fluxes, axis=0), 1e-6)
    htr_mean = np.nanmean(train_htr, axis=0)
    htr_std = np.maximum(np.nanstd(train_htr, axis=0), 1e-6)
    invalid_htr_levels = np.flatnonzero(
        ~np.isfinite(htr_mean) | ~np.isfinite(htr_std)
    )
    if invalid_htr_levels.size:
        raise ValueError(
            "Cannot normalize heating rates; non-finite training statistics at "
            f"levels {invalid_htr_levels.tolist()}"
        )

    train_data = RRTMGDataset(
        train_fluxes, train_htr, flux_mean, flux_std, htr_mean, htr_std
    )
    validation_data = RRTMGDataset(
        test_fluxes,
        test_htr,
        flux_mean,
        flux_std,
        htr_mean,
        htr_std,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(
        args.model,
        train_fluxes.shape[1],
        train_htr.shape[1],
        args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    best_validation_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"Checkpoint model is {checkpoint['model_name']!r}, not {args.model!r}"
            )
        if args.model == "fnn" and checkpoint.get("hidden_layers", [128, 128]) != args.hidden_layers:
            raise ValueError(
                f"Checkpoint hidden layers are {checkpoint.get('hidden_layers', [128, 128])}, "
                f"not {args.hidden_layers}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        training_loss = 0.0
        for inputs, targets in tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{args.epochs} train", leave=False
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs).squeeze(-1)
            loss = masked_mse(predictions, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(inputs)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(
                validation_loader,
                desc=f"epoch {epoch + 1}/{args.epochs} validation",
                leave=False,
            ):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                predictions = model(inputs).squeeze(-1)
                validation_loss += masked_mse(predictions, targets).item() * len(inputs)

        training_loss /= len(train_data)
        validation_loss /= len(validation_data)
        is_best = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss, validation_loss)
        state = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "flux_mean": flux_mean,
            "flux_std": flux_std,
            "htr_mean": htr_mean,
            "htr_std": htr_std,
            "num_bands": train_fluxes.shape[1],
            "num_levels": train_htr.shape[1],
            "hidden_layers": args.hidden_layers if args.model == "fnn" else None,
        }
        checkpoint_name = args.model
        if args.model == "fnn":
            checkpoint_name += "_" + "x".join(map(str, args.hidden_layers))
        save_checkpoint(args.checkpoint_dir / f"{checkpoint_name}_latest.pt", state)
        if is_best:
            save_checkpoint(args.checkpoint_dir / f"{checkpoint_name}_best.pt", state)

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train={training_loss:.6f} validation={validation_loss:.6f}"
        )
        experiment.log_metrics(
            {"train_loss": training_loss, "validation_loss": validation_loss},
            epoch=epoch + 1,
        )

    experiment.end()


if __name__ == "__main__":
    main()
