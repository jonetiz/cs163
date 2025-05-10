# Changes in Socioeconomic Classes Over Time
A Data Science project by Henry Chau and Jonathan Etiz.

This project is an exploration into earnings data to see the effect of natural inflation on socioeconomic classes, particularly how the lower, middle, and upper classes (as defined by Pew Research Center) respond in different ways.

## Application Setup
The website is served by [Plotly Dash](https://dash.plotly.com/)
### Requirements
- Python >= 3.10

### Install Dependencies
In the project directory, execute this command to automatically install the necessary Python dependencies:
```
pip install -r requirements.txt
```

### Data Processing
The [data processing notebook](https://www.github.com/jonetiz/cs163/tree/main/data/data_processing.ipynb) contains step-by-step instructions to process the data to a practical size that is used on the web app.

### Local Setup (Development Environment)
After installing dependencies and processing the data, the project can be run in a local development environment by executing this command from the project directory:
```
python cs163website.py
```

### Google App Engine Setup
TODO

## Data Pipeline
The data used in this project is from the [US Census Bureau's Annual Social and Economic Supplements](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html) between the years of 2014 and 2024. The data is combined with many (unused) features dropped, and then uploaded to a Google Cloud Storage bucket.

All visualizations are rendered and analyses performed on Google App Engine from the merged, culled datset.

## Project Structure
```
./
├─ assets/                      # Static assets
│   └─ stylesheet.css
│
├─ data/                        # Files pertinent data processing
│   ├─ analysis.ipynb           # Analysis & visualization code
│   └─ data_processing.ipynb    # Processes original dataset
│
├─ pages/                       # Plotly dash pages
│   ├─ findings.py
│   ├─ home.py
│   ├─ methodology.py
│   └─ objectives.py
│
├─ .gcloudignore                # Google App Engine ignore file
├─ .gitignore
├─ analysis.py                  # Analysis & visualization code for Plotly dash
├─ app.yaml.template            # Google App Engine app.yaml template file
├─ cs163website.py              # Plotly dash main file
├─ data.py                      # Module for loading dataset to Plotly dash
├─ README.md
└─ requirements.txt
````

## Website Link
The website can be found at https://cs-163-452620.wl.r.appspot.com/.