# Neural Fractal

An experiment in teaching a neural network to draw fractals, made for learning and artistic purposes.

It's a neural fractal approximator built with [PyTorch](https://pytorch.org/) that learns fractals such as **Mandelbrot** and **Julia**. Images are rendered progressively during training, so you can watch the fractal take shape in real time. 

<img src="fractals/julia.png" alt="Julia set" width="128"> <img src="fractals/mandelbrot.png" alt="Mandelbrot set" width="128"> <img src="fractals/burning_ship.png" alt="Burning Ship fractal" width="128"> <img src="fractals/newton.png" alt="Newton fractal" width="128">

## Features

- Four fractal presets: Mandelbrot, Julia, Burning Ship and Newton
- Live image updates every epoch, reflecting the training process as it happens
- Full YAML configuration for model, training and fractal parameters
- Jupyter notebook for visualizing overall convergence quality

## Requirements

- Python 3.10+
- PyTorch
- Pillow
- PyYAML
- pandas
- Matplotlib
- Jupyter

## Installation

```bash
pip install -r requirements.txt
```

## How the Network Learns a Fractal

The model is a function from a point to a color. It never sees the fractal's formula, only points and their true colors, a bit like learning a map by being dropped to random places and then drawing the whole picture from memory, 

1. **Sample:** Random `(x, y)` points in the square from -2 to 2 are drawn once at the start (`samples`) and reshuffled every epoch.
   
2. **Compute:** `architecture/targets.py` runs the fractal's iteration on each point for up to `iters` steps and counts how many steps the point stays bounded (Julia, Mandelbrot, Burning Ship) or how many it spends settled on a root (Newton). That count is divided by `iters` and passed through sine waves (the `cmap` values in `configs/fractal.yaml`) to get an RGB color.
   
3. **Encode:** Each point is expanded with sine and cosine waves at `frequencies` increasing speeds, giving `2 + 4*frequencies` inputs (26 with the default of 6). Raw coordinates change slowly across the image, so on their own they tend to produce smooth, blurry output. Low frequencies capture coarse position while high ones separate nearby pixels, which lets the network pick up repeating patterns.
   
4. **Predict:** A fully connected network (`layers` hidden layers of `hidden_dim` ReLU units) maps the encoded point to three outputs: red, green and blue.
   
5. **Epoch:** The prediction is scored against the true color with mean squared error, and the weights are adjusted with Adam.

## Configuration

Most parameters live in two YAML files:

- `configs/default.yaml`: model, dataset, training and generation settings, including which fractal preset to use.
- `configs/fractal.yaml`: the parameters for each fractal preset (iteration count, constants, color mapping).

The defaults train on the Julia preset for 30 epochs and render at 512x512:

```yaml
model:
  frequencies: 6
  hidden_dim: 256
  num_layers: 4

dataset:
  num_samples: 102400
  batch_size: 1024

training:
  epochs: 30
  lr: 0.001

generation:
  resolution: 512
  preset: "julia" # julia | mandelbrot | burning_ship | newton

paths:
  live: "live/"
  log: "log/"
```

- Keep the trailing slash on the `paths` entries, since file names are appended to them directly.
- `batch_size` does not necessarily need to be in relation to `num_samples`.

## Training & Results

```bash
python train.py
```

- `live/fractal.png` is overwritten after every epoch with the model's current attempt.
- `log/epoch_{n}.png` is saved every 10 epochs and on the final epoch, so you can compare snapshots later.
- `log/log.csv` records the loss for each epoch and is written when training finishes.

Open `analysis.ipynb` to look at the logged data together.

## License

[MIT license](LICENSE)
