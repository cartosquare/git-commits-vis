# Terrain classify server

## Requirements

The demo server requires Python with some dependencies.
To make sure you have the dependencies, please run `pip install -r examples/web_demo/requirements.txt`, and also make sure that you've compiled the Python Caffe interface and that it is on your `PYTHONPATH` (see [installation instructions](/installation.html)).

## Run

Running `python examples/web_demo/app.py` will bring up the demo server, accessible at `http://0.0.0.0:5000`.
You can enable debug mode of the web server, or switch to a different port:

    % python examples/web_demo/app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on

## You may need to set follow environment
```
export HDF5_DISABLE_VERSION_CHECK=1
```
