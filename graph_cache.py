import plotly.io as pio
import json, os

os.makedirs('cache', exist_ok=True)

def cache(figure, file_name):
    """Saves a Plotly `Figure` object in the cache with the specified `file_name`."""
    
    pio.write_json(figure, f'cache/{file_name}.json')

def get(file_name):
    """Returns a plotly figure object cached with the specified `file_name`."""

    return pio.read_json(f'cache/{file_name}.json')