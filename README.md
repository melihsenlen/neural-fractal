# neural_fractal

A small experiment in teaching a neural network to draw fractals. It's a neural fractal approximator built with [PyTorch](https://pytorch.org/) that learns fractals such as **Mandelbrot** and **Julia**. Images are rendered progressively during training, so you can watch the fractal take shape in real time.

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
- Matplotlib
- pandas

Training runs on a CUDA GPU if one is available and falls back to the CPU otherwise.

## Installation

```bash
pip install -r requirements.txt
```

## How the Network Learns a Fractal

The model is a function from a point to a color. It never sees the fractal's formula, only points and their true colors, a bit like learning a map by being dropped to random places and then drawing the whole thing from memory.

Each run goes through the same pipeline:

1. **Sample points.** Random `(x, y)` points in the square from -2 to 2 are drawn once at the start (`num_samples`) and reshuffled every epoch.
   
2. **Compute the true color.** `architecture/targets.py` runs the fractal's iteration on each point for up to `iters` steps and counts how many steps the point stays bounded (Julia, Mandelbrot, Burning Ship) or how many it spends settled on a root (Newton). That count is divided by `iters` and passed through sine waves (the `cmap` values in `configs/fractal.yaml`) to get an RGB color.
   
3. **Encode the input.** Each point is expanded with sine and cosine waves at `num_frequencies` evenly spaced frequencies, giving `2 + 4 * num_frequencies` inputs (26 with the default of 6). Raw coordinates alone tend to produce smooth, blurry output, and the waves are what lets the network pick up fine, repeating detail.
   
4. **Predict a color.** A fully connected network (`num_layers` hidden layers of `hidden_dim` ReLU units) maps the encoded point to three outputs: red, green and blue.
   
5. **Update.** The prediction is scored against the true color with mean squared error, and the weights are adjusted with Adam Optimizer.
    
6. **Render.** After every epoch, the network is queried at every pixel of a `resolution` x `resolution` grid and the result is saved as `live/fractal.png`. This is the network's current guess, not the formula's raw output.

## Configuration

Most parameters live in two YAML files:

- `configs/default.yaml`: model, dataset, training and generation settings, including which fractal preset to use.
- `configs/fractal.yaml`: the parameters for each fractal preset (iteration count, constants, color mapping).

The defaults train on the Julia preset for 30 epochs and render at 512x512:

```yaml
model:
  num_frequencies: 6
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
  preset: "julia"   # julia | mandelbrot | burning_ship | newton

paths:
  live: "live/"
  log: "log/"
```

To train on a different fractal, change `generation.preset`. Keep the trailing slash on the `paths` entries, since file names are appended to them directly.

## Training and Results

Start training with:

```bash
python train.py
```

Output is written as it goes:

- `live/fractal.png` is overwritten after every epoch with the model's current attempt.
- `log/epoch_<N>.png` is saved every 10 epochs and on the final epoch, so you can compare snapshots later.
- `log/log.csv` records the loss for each epoch and is written when training finishes.

Open `analysis.ipynb` to look at the logged data.

## License

MIT License
