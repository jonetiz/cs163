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
The application runs on [Google App Engine](https://cloud.google.com/appengine?hl=en), with data stored on [Google Cloud Storage (GCS)](https://cloud.google.com/storage?hl=en). To deploy the production environment to Google Cloud, create a project in the Google Cloud console, and upload the files `merged_asec.csv` and `merged_fam.csv`, that were generated in the **Data Processing** step above to the GCS bucket for the project.

Make a copy of the `app.yaml.template` file in the root directory of this project, and supplant the `INSERT_GCLOUD_BUCKET_NAME` string with the name of your project's gcloud bucket.

Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) on your local machine, and run the `gloud init` command to login and set the project.

Once you have logged in, set the correct project, and configured `app.yaml`, execute the following command in the project directory:
```
gcloud app deploy
```

## Data Pipeline
The data used in this project is from the [US Census Bureau's Annual Social and Economic Supplements](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html) between the years of 2014 and 2024. The data is combined with many (unused) features dropped, and then uploaded to a Google Cloud Storage bucket.

All visualizations are rendered and analyses performed on Google App Engine from the merged, culled datasets, which are stored on GCS.

## Project Structure
```
./
├─ assets/                      # Static assets
│   └─ stylesheet.css
│
├─ data/
│   └─ data_processing.ipynb    # Instructions for data pre-processing
│
├─ pages/                       # Plotly dash pages; these do not have logic other than page formatting
│   ├─ findings.py
│   ├─ home.py
│   ├─ methodology.py
│   └─ objectives.py
│
├─ .gcloudignore                # Google App Engine ignore file
├─ .gitignore
├─ analysis.py                  # Analysis & visualization code - all logic and visualizations are here
├─ app.yaml.template            # Google App Engine app.yaml template file
├─ cs163website.py              # Plotly dash main file
├─ data.py                      # Module for loading dataset to Plotly dash
├─ README.md
└─ requirements.txt
```

## Website Link
The website can be found at https://cs-163-452620.wl.r.appspot.com/.