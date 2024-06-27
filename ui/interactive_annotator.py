import dash
from dash import html, dcc, Input, Output, State, clientside_callback
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import base64
from io import BytesIO
import os
import json
from dash.exceptions import PreventUpdate
from functools import lru_cache

# Global paths to the image and scores folders
NPY_FOLDER_PATH = '/mnt/disks/whole/whole-slides/'
NPY_FOLDER_POS_PATH = '/home/heguanglin/data/061924/parasite-slides-cleaned/'
SCORES_FOLDER_PATH = '/home/heguanglin/octopi/out/resnet18/h7h28_v3_h7_v4/csv'

# Number of items per page
ITEMS_PER_PAGE = 250
BAR_PLOT_HEIGHT = 700

# Initialize the Flask app
server = Flask(__name__)
auth = HTTPBasicAuth()

# User credentials
users = {
    "cephla": generate_password_hash("octopi")
}

@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False

# Middleware to protect Dash routes
@server.before_request
def before_request_func():
    if not request.endpoint or request.endpoint == 'static':
        return
    return auth.login_required(lambda: None)()

# Initialize the Dash app
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

# Caching function for image conversion
@lru_cache(maxsize=1000)
def numpy_array_to_image_string(frame_tuple):
    frame = np.array(frame_tuple)
    frame = frame.transpose(1, 2, 0)
    img_fluorescence = frame[:, :, [2, 1, 0]]
    img_dpc = frame[:, :, 3]
    img_dpc = np.dstack([img_dpc, img_dpc, img_dpc])
    img_overlay = 0.64 * img_fluorescence + 0.36 * img_dpc
    frame = img_overlay.astype('uint8')
    img = Image.fromarray(frame, 'RGB')
    with BytesIO() as buffer:
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()

# Get the list of slides from the scores folder
slide_files = [f.replace('.csv', '') for f in os.listdir(SCORES_FOLDER_PATH) if f.endswith('.csv')]

labels_df = pd.read_csv('utils/label.csv')
def _get_label(file_name):
    name = file_name
    new_label_row = labels_df[labels_df['name'] == name]
    if not new_label_row.empty:
        return new_label_row['new label'].values[0]
    else:
        return name

labels = [_get_label(slide) for slide in slide_files]

# App layout
app.layout = html.Div([
    html.Div([
        html.Label('Select Slide:', style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='slide-dropdown',
            options=[{'label': label, 'value': slide} for label, slide in zip(labels, slide_files)],
            style={'width': '300px'},
            clearable=False
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
    html.Div(id='slide-info', style={'textAlign': 'center'}),
    html.Div([
        "Page: ",
        dcc.Input(id='page-input', type='number', value=1, min=1, style={'width': '100px'}),
    ]),
    dcc.RadioItems(
        id='sort-order',
        options=[
            {'label': 'Ascending', 'value': 'asc'},
            {'label': 'Descending', 'value': 'desc'},
        ],
        value='desc'
    ),
    html.Div([
        html.Label('Create Class:', style={'marginRight': '10px'}),
        dcc.Input(id='class-name', type='text', placeholder='Class Name', style={'marginRight': '10px'}),
        dcc.Input(id='class-color', type='text', placeholder='green', style={'marginRight': '10px'}),
        html.Button('Add Class', id='add-class-btn', n_clicks=0)
    ],  style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
    html.Div(id='class-panel', style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
    html.Button('Annotate', id='annotate-btn',style={'height':'50px', 'width':'100px','marginRight': '20px'}),
    html.Button('Export Annotations', id='export-btn', n_clicks=0,style={'height':'50px', 'width':'100px'}),
    dcc.Download(id='download-dataframe-csv'),
    html.Div(id='image-grid'),
    dcc.Store(id='annotations-store', data={}),
    dcc.Store(id='selected-indices-store', data=[]),
    dcc.Store(id='classes-store', data=[
        {'name': 'parasite', 'color': '#ff0000'},
        {'name': 'non-parasite', 'color': '#0000ff'}
    ])
])

# Load the numpy data and CSV data once
npy_data_cache = {}
csv_data_cache = {}

@app.callback(
    [Output('slide-info', 'children'),
     Output('image-grid', 'children')],
    [Input('slide-dropdown', 'value'),
     Input('page-input', 'value'),
     Input('sort-order', 'value')],
    prevent_initial_call=True
)
def update_images(slide, page_number, sort_order):
    if slide is None:
        return "No slide selected", html.Div()

    # Load data from cache or file
    if slide not in npy_data_cache:
        npy_data_cache[slide] = np.load(os.path.join(NPY_FOLDER_PATH, slide + '.npy'))
        csv_data_cache[slide] = pd.read_csv(os.path.join(SCORES_FOLDER_PATH, slide + '.csv'))

        pos_files = [f for f in os.listdir(NPY_FOLDER_POS_PATH) if f.startswith(slide)]
        if pos_files:
            npy_data_cache[slide] = np.load(os.path.join(NPY_FOLDER_POS_PATH, pos_files[0]))

        assert len(npy_data_cache[slide]) == len(csv_data_cache[slide])

    npy_data = npy_data_cache[slide]
    csv_data = csv_data_cache[slide]

    index = csv_data['index']
    scores = csv_data['parasite output']

    # Sort the scores and index
    sorted_indices = scores.sort_values(ascending=(sort_order == 'asc')).index
    sorted_scores = scores[sorted_indices]
    sorted_data = npy_data[index[sorted_indices]]

    # Pagination logic
    start_index = (page_number - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_data = sorted_data[start_index:end_index]
    paginated_scores = sorted_scores[start_index:end_index]

    # Convert numpy arrays to images and encode them
    image_elements = []
    for i, (arr, score) in enumerate(zip(paginated_data, paginated_scores), start=start_index + 1):
        encoded_image = numpy_array_to_image_string(tuple(map(tuple, arr)))
        image_html = html.Img(src=f"data:image/png;base64,{encoded_image}", style={'height': '124px', 'width': '124px'}, id={'type': 'image', 'index': i})
        score_html = html.Div(f"{score:.2f}", style={'textAlign': 'center'})
        number_html = html.Div(f"[{i}]", style={'textAlign': 'center', 'color': 'gray'})
        image_elements.append(html.Div([number_html, image_html, score_html], style={'margin': '10px', 'display': 'inline-block'}))

    return f"Slide {_get_label(slide)}: {slide}; Images count:{len(npy_data)}", html.Div(image_elements)

# Clientside callback for image selection
app.clientside_callback(
    """
    function(n_clicks, selectedIndices) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return {};

        const prop_id = ctx.triggered[0]['prop_id'];
        const index = JSON.parse(prop_id.split('.')[0])['index'];
        
        let newSelectedIndices = [...selectedIndices];
        const indexPosition = newSelectedIndices.indexOf(index);
        
        if (indexPosition > -1) {
            newSelectedIndices.splice(indexPosition, 1);
        } else {
            newSelectedIndices.push(index);
        }

        return [newSelectedIndices, {[index]: {'boxShadow': indexPosition > -1 ? 'none' : '0 0 10px 10px grey'}}];
    }
    """,
    [Output('selected-indices-store', 'data'),
     Output({'type': 'image', 'index': dash.ALL}, 'style')],
    [Input({'type': 'image', 'index': dash.ALL}, 'n_clicks')],
    [State('selected-indices-store', 'data')]
)

@app.callback(
    Output('annotations-store', 'data'),
    [Input('annotate-btn', 'n_clicks')],
    [State('selected-indices-store', 'data'),
     State('class-panel', 'children'),
     State('annotations-store', 'data')]
)
def update_annotations(n_clicks, selected_indices, class_panel, current_annotations):
    if n_clicks == 0:
        raise PreventUpdate

    if class_panel:
        selected_class = class_panel[0]['props']['children'][1]['props']['value']
        for index in selected_indices:
            current_annotations[str(index)] = selected_class

    return current_annotations

@app.callback(
    Output('page-input', 'value'),
    [Input('slide-dropdown', 'value')],
    [State('page-input', 'value')]
)
def reset_page_number(slide, current_page):
    return 1 if slide else current_page

@app.callback(
    Output('class-panel', 'children'),
    [Input('add-class-btn', 'n_clicks'),
     Input('classes-store', 'data')],
    [State('class-name', 'value'),
     State('class-color', 'value')]
)
def update_class_panel(n_clicks, classes, class_name, class_color):
    ctx = dash.callback_context
    if ctx.triggered_id == 'add-class-btn' and class_name and class_color:
        classes.append({'name': class_name, 'color': class_color})

    class_elements = [
        html.Div([
            html.Label('Class: ', style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='class-dropdown',
                options=[{'label': c['name'], 'value': c['color']} for c in classes],
                value=classes[0]['color'],
                style={'width': '400px'}
            )
        ], style={'marginRight': '20px'}),
        html.Div(
            [html.Span(f"{c['name']} ", style={'color': c['color'], 'marginRight': '10px', 'border': '1px solid', 'padding': '5px'}) for c in classes],
            style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
        )
    ]
    
    return class_elements

@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('export-btn', 'n_clicks')],
    [State('annotations-store', 'data')],
    prevent_initial_call=True
)
def export_annotations(n_clicks, annotations):
    if n_clicks > 0:
        df = pd.DataFrame(list(annotations.items()), columns=['Index', 'Class'])
        return dcc.send_data_frame(df.to_csv, 'annotations.csv')

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)