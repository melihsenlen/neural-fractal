import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from architecture.config import load_config, path
from architecture.model import NeuralFractal
from architecture.targets import targets


def render(model: torch.nn.Module, resolution: int, device: str) -> Image.Image:
    x = torch.linspace(-2, 2, resolution)
    y = torch.linspace(-2, 2, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    with torch.no_grad():
        rgb = model(coords).clamp(0, 1).cpu().numpy()
    return Image.fromarray((rgb.reshape(resolution, resolution, 3) * 255).astype("uint8"))


if __name__ == "__main__":
    defaults = load_config("configs/default.yaml")
    fractals = load_config("configs/fractal.yaml")

    model_params   = defaults["model"]
    dataset_params = defaults["dataset"]

    training   = defaults["training"]
    generation = defaults["generation"]
    paths      = defaults["paths"]

    fractal    = fractals[generation["preset"]]
    resolution = generation["resolution"]

    dataloader = DataLoader(
        (torch.rand(dataset_params["num_samples"], 2) - 0.5) * 4, # dataset
        dataset_params["batch_size"], # batch_size
        True # shuffle
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralFractal(
        model_params["frequencies"],
        model_params["hidden_dim"],
        model_params["num_layers"]
    ).to(device)

    live_path = path(defaults, "live")
    live_path.mkdir(exist_ok=True)
    log_path  = path(defaults, "log")
    log_path.mkdir(exist_ok=True)

    epochs    = training["epochs"]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), training["lr"])

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for coords in dataloader:
            coords = coords.to(device)
            target = targets(coords, fractal).to(device)
            output = model(coords)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch: {epoch + 1:>2}/{epochs} | Loss: {avg_loss:8.6f}")

        img = render(model, resolution, device)
        img.save(live_path / "fractal.png")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            img.save((log_path / f"epoch_{epoch + 1}.png"))
    
    pd.Series(
        data=loss_history, name="loss", # data
        index=pd.RangeIndex(1, (len(loss_history) + 1), name="epoch") # index
    ).to_csv((log_path / "log.csv"))