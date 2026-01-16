# Aerosense Layout Designer

Interactive Streamlit app for designing sensor layouts on blade geometries.

## Installation

This project uses `uv` for dependency management. If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project dependencies:

```bash
uv sync
```

## Running the App

Start the Streamlit app with:

```bash
uv run streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. Upload a blade geometry file (.dat or .txt) using the sidebar
2. Configure the wing dimensions (chord length in meters)
3. Set up sensor patches:
   - Number of patches
   - Position along the chord (0.0 to 1.0)
   - Surface side (pressure or suction)
   - Reference sensor index (0-23)
4. Click "Generate Layout" to visualize the sensor arrangement
5. Download the resulting image if needed

## File Format

The blade geometry file should contain airfoil coordinates in a standard format (x,y pairs representing the airfoil shape).
