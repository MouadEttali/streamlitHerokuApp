from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load


app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html',href="static/base_pic.svg")
    else:
        text = request.form['text']
        path = "static/pred_pic.svg"
        model = load('model.joblib')
        np_array = floats_string_to_input_arr(text)
        make_picture('AgesAndHeights.pkl', np_array, model, path)
        return render_template('index.html', href=path)


def make_picture(training_data_filename, new_input_arr, model, output_file='predictions_pic.svg'):
    # Plot training data with model
    data = pd.read_pickle(training_data_filename)
    x_new = np.arange(18).reshape((18, 1))
    preds  = model.predict(x_new)
    ages = data['Age']
    heights = data['Height']
    fig = px.scatter(x=ages, y=heights, title="Height vs Age", labels={'x': 'Age (Years)',
                                                                    'y': 'Height (Inches)'})
    fig.add_trace(
        go.Scatter(x=x_new.reshape(x_new.shape[0]), y=preds, mode='lines', name='Model'))

    if new_input_arr is not False:
        # Plot new predictions
        new_preds = model.predict(new_input_arr)
        fig.add_trace(
        go.Scatter(x=new_input_arr.reshape(new_input_arr.shape[0]), y=new_preds, name='New Outputs', mode='markers', marker=dict(
                color='purple',
                size=20,
                line=dict(
                    color='purple',
                    width=2
                ))))
    fig.write_image(output_file, width=800,engine="kaleido")
    return fig

def floats_string_to_input_arr(floats_str):
    floats = [float(x.strip()) for x in floats_str.split(',')]
    as_np_arr = np.array(floats).reshape(len(floats), 1)
    return as_np_arr