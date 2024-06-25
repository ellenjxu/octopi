import dash
from dash import html, dcc, Input, Output, State, callback_context
import pandas as pd
import numpy as np
from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import base64
from io import BytesIO
import os
import time
import logging
import json
from dash.exceptions import PreventUpdate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global paths
NPY_FOLDER_PATH = '/mnt/disks/whole/whole-slides/'
NPY_FOLDER_POS_PATH = '/home/heguanglin/data/061924/parasite-slides-cleaned/'
SCORES_FOLDER_PATH = '/home/heguanglin/octopi/out/resnet18/h7h28_v3_h7_v4/csv'

# Initialize Flask app and authentication
server = Flask(__name__)
auth = HTTPBasicAuth()
users = {"cephla": generate_password_hash("octopi")}

@auth.verify_password
def verify_password(username, password):
    return username in users and check_password_hash(users.get(username), password)

@server.before_request
def before_request_func():
    if not request.endpoint or request.endpoint == 'static':
        return
    return auth.login_required(lambda: None)()

# Initialize Dash app
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

def numpy_array_to_image_string(frame):
    frame = frame.transpose(1, 2, 0)
    img_fluorescence = frame[:, :, [2, 1, 0]]
    img_dpc = frame[:, :, 3]
    img_overlay = (0.64 * img_fluorescence + 0.36 * np.dstack([img_dpc]*3)).astype('uint8')
    img = Image.fromarray(img_overlay, 'RGB')
    with BytesIO() as buffer:
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

# Get slide files and labels
slide_files = [f.replace('.csv', '') for f in os.listdir(SCORES_FOLDER_PATH) if f.endswith('.csv')]
labels_df = pd.read_csv('utils/label.csv')
def _get_label(file_name):
    new_label_row = labels_df[labels_df['name'] == file_name]
    return new_label_row['new label'].values[0] if not new_label_row.empty else file_name
labels = [_get_label(slide) for slide in slide_files]

# App layout
app.layout = html.Div([
    html.H1("Image Annotation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.Label('Select Slide:', style={'marginRight': '10px'}),
            dcc.Dropdown(id='slide-dropdown', options=[{'label': label, 'value': slide} for label, slide in zip(labels, slide_files)],
                         style={'width': '300px'}, clearable=False)
        ], style={'marginRight': '20px'}),
        html.Div([
            html.Label("Images per page: ", style={'marginRight': '10px'}),
            dcc.Input(id='items-per-page', type='number', value=100, min=1, max=500, step=1, style={'width': '60px'}),
        ], style={'marginRight': '20px'}),
        html.Div([
            html.Label("Page: ", style={'marginRight': '10px'}),
            dcc.Input(id='page-input', type='number', value=1, min=1, style={'width': '60px'}),
        ]),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.Label("Sort Order: ", style={'marginRight': '10px'}),
            dcc.RadioItems(id='sort-order', options=[{'label': 'Ascending', 'value': 'asc'}, {'label': 'Descending', 'value': 'desc'}],
                           value='desc', inline=True)
        ], style={'marginRight': '20px'}),
        html.Div(id='slide-info', style={'fontWeight': 'bold'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.Label('Create New Class:', style={'marginRight': '10px'}),
            dcc.Input(id='class-name', type='text', placeholder='Class Name', style={'marginRight': '10px'}),
            dcc.Input(id='class-color', type='text', placeholder='Color (e.g., #FF0000)', style={'marginRight': '10px'}),
            html.Button('Add Class', id='add-class-btn', n_clicks=0, style={'marginRight': '20px'}),
        ], style={'marginBottom': '10px'}),
        html.Div(id='class-buttons', style={'marginBottom': '10px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Button('Export Annotations', id='export-btn', n_clicks=0, 
                    style={'height':'40px', 'width':'150px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    
    html.Div(id='image-grid', style={'textAlign': 'center'}),
    
    dcc.Download(id='download-dataframe-csv'),
    dcc.Store(id='annotations-store'),
    dcc.Store(id='classes-store', data=[{'name': 'parasite', 'color': '#ff0000'}, {'name': 'non-parasite', 'color': '#0000ff'}]),
    dcc.Store(id='selected-class-store', data=None),
    dcc.Store(id='client-side-store', storage_type='local'),  # Add this line for client-side storage
    html.Div(id='debug-output', style={'marginTop': '20px', 'textAlign': 'center'})
])

@app.callback(
    [Output('slide-info', 'children'), Output('image-grid', 'children'), Output('debug-output', 'children')],
    [Input('slide-dropdown', 'value'), Input('page-input', 'value'), Input('sort-order', 'value'), Input('items-per-page', 'value')],
    [State('annotations-store', 'data'), State('classes-store', 'data'), State('client-side-store', 'data')]
)
def update_images(slide, page_number, sort_order, items_per_page, annotations, classes, client_side_data):
    start_time = time.time()
    logging.info(f"Starting update_images for slide: {slide}, page: {page_number}, sort_order: {sort_order}, items_per_page: {items_per_page}")

    if slide is None:
        return "No slide selected", html.Div(), "No slide selected"

    try:
        csv_data = pd.read_csv(os.path.join(SCORES_FOLDER_PATH, slide + '.csv'))
        csv_data = csv_data.sort_values('parasite output', ascending=(sort_order == 'asc'))
        
        start_index = (page_number - 1) * items_per_page
        end_index = start_index + items_per_page
        page_data = csv_data.iloc[start_index:end_index]

        npy_data = np.load(os.path.join(NPY_FOLDER_PATH, slide + '.npy'))
        pos_files = [f for f in os.listdir(NPY_FOLDER_POS_PATH) if f.startswith(slide)]
        if pos_files:
            npy_data = np.load(os.path.join(NPY_FOLDER_POS_PATH, pos_files[0]))

        annotations = client_side_data.get(slide, {}) if client_side_data else {}
        class_colors = {c['name']: c['color'] for c in classes}

        image_elements = []
        for _, row in page_data.iterrows():
            img_index = int(row['index'])
            score = row['parasite output']
            
            encoded_image = numpy_array_to_image_string(npy_data[img_index])
            
            image_html = html.Img(src=f"data:image/png;base64,{encoded_image}", 
                                  style={'height': '124px', 'width': '124px'}, 
                                  id={'type': 'image', 'index': img_index})
            
            annotation_class = annotations.get(str(img_index), '')
            annotation_color = class_colors.get(annotation_class, 'transparent')
            
            annotation_div = html.Div(style={
                'position': 'absolute', 
                'top': '0', 
                'left': '0', 
                'width': '100%', 
                'height': '100%', 
                'border': f'4px solid {annotation_color}',
                'boxSizing': 'border-box',
                'pointerEvents': 'none'
            }, id={'type': 'annotation', 'index': img_index})
            
            score_html = html.Div(f"{score:.2f}", style={'textAlign': 'center'})
            number_html = html.Div(f"[{img_index}]", style={'textAlign': 'center', 'color': 'gray'})
            
            image_container = html.Div([
                html.Div([image_html, annotation_div], style={'position': 'relative'}),
                score_html,
                number_html
            ], style={'margin': '10px', 'display': 'inline-block'})
            
            image_elements.append(image_container)

        processing_time = time.time() - start_time
        logging.info(f"Processing completed in {processing_time:.2f} seconds")

        debug_info = f"Processed {len(image_elements)} images in {processing_time:.2f} seconds"
        return f"Slide {_get_label(slide)}: {slide}; Total Images: {len(csv_data)}", html.Div(image_elements), debug_info

    except Exception as e:
        logging.error(f"Error in update_images: {str(e)}")
        return f"Error loading data for slide {slide}", html.Div(), f"Error: {str(e)}"

@app.callback(
    [Output({'type': 'annotation', 'index': dash.dependencies.ALL}, 'style'), 
     Output('annotations-store', 'data'),
     Output('client-side-store', 'data')],
    [Input({'type': 'image', 'index': dash.dependencies.ALL}, 'n_clicks'), 
     Input('selected-class-store', 'data'),
     Input('slide-dropdown', 'value')],
    [State({'type': 'image', 'index': dash.dependencies.ALL}, 'id'), 
     State('annotations-store', 'data'), 
     State('classes-store', 'data'),
     State('client-side-store', 'data')]
)
def update_annotation(n_clicks, selected_class, current_slide, image_ids, annotations, classes, client_side_data):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    client_side_data = client_side_data or {}
    annotations = client_side_data.get(current_slide, {})
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'index' in triggered_id and selected_class is not None:
        clicked_index = json.loads(triggered_id)['index']
        if str(clicked_index) in annotations and annotations[str(clicked_index)] == selected_class:
            del annotations[str(clicked_index)]
        else:
            annotations[str(clicked_index)] = selected_class

    class_colors = {c['name']: c['color'] for c in classes}
    new_styles = [
        {
            'position': 'absolute', 
            'top': '0', 
            'left': '0', 
            'width': '100%', 
            'height': '100%', 
            'border': f"4px solid {class_colors.get(annotations.get(str(img_id['index']), ''), 'transparent')}",
            'boxSizing': 'border-box',
            'pointerEvents': 'none'
        }
        for img_id in image_ids
    ]

    # Update client-side storage
    client_side_data[current_slide] = annotations

    return new_styles, annotations, client_side_data

@app.callback(
    Output('page-input', 'value'),
    [Input('slide-dropdown', 'value')],
    [State('page-input', 'value')]
)
def reset_page_number(slide, current_page):
    return 1 if slide else current_page

@app.callback(
    [Output('class-buttons', 'children'), Output('classes-store', 'data')],
    [Input('add-class-btn', 'n_clicks')],
    [State('class-name', 'value'), State('class-color', 'value'), State('classes-store', 'data')]
)
def update_class_buttons(n_clicks, class_name, class_color, classes):
    if n_clicks > 0 and class_name and class_color:
        classes.append({'name': class_name, 'color': class_color})

    class_buttons = [
        html.Button(c['name'], id={'type': 'class-button', 'index': i},
                    style={'backgroundColor': c['color'], 'color': 'white', 'margin': '5px'})
        for i, c in enumerate(classes)
    ]
    
    return class_buttons, classes

@app.callback(
    Output('selected-class-store', 'data'),
    [Input({'type': 'class-button', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('classes-store', 'data')]
)
def update_selected_class(n_clicks, classes):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_index = json.loads(button_id)['index']
    return classes[button_index]['name']

@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('export-btn', 'n_clicks')],
    [State('client-side-store', 'data')],
    prevent_initial_call=True
)
def export_annotations(n_clicks, client_side_data):
    if n_clicks > 0 and client_side_data:
        all_annotations = []
        for slide, annotations in client_side_data.items():
            for index, class_name in annotations.items():
                all_annotations.append({'Slide': slide, 'Index': index, 'Class': class_name})
        df = pd.DataFrame(all_annotations)
        return dcc.send_data_frame(df.to_csv, 'annotations.csv')
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True, port=8055) 