import torch
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path

from architecture.config import load_config
from architecture.model import FractalNet
from architecture.targets import targets


def render_fractal(model: torch.nn.Module, resolution: int, device: str) -> Image.Image:
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

    fractal = fractals[generation["preset"]]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    dataset    = (torch.rand(dataset_params["num_samples"], 2) - 0.5) * 4
    dataloader = DataLoader(dataset, batch_size=dataset_params["batch_size"], shuffle=True)

    model = FractalNet(
        hidden_dim=model_params["hidden_dim"],
        num_layers=model_params["num_layers"],
        num_frequencies=model_params["num_frequencies"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    criterion = torch.nn.MSELoss()

    live_path = paths["live"]
    Path(live_path).mkdir(exist_ok=True)
    log_path  = paths["log"]
    Path(log_path).mkdir(exist_ok=True)

    resolution = generation["resolution"]
    epochs     = training["epochs"]

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for coords in dataloader:
            coords = coords.to(device)
            target = targets(coords, fractal).to(device)

            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

        img = render_fractal(model, resolution=resolution, device=device)
        img.save(Path(live_path + "fractal.png"))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            img.save(Path(log_path + f"epoch_{epoch + 1}.png"))

    with open(Path(log_path + "log.csv"), "w") as c:
        c.write("epoch,loss\n")
        for i, loss in enumerate(loss_history):
            c.write(f"{i + 1},{loss:.6f}\n")