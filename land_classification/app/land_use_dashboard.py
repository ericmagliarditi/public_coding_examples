import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import landpy
import os
import numpy as np
import random
import glob
import plotly.express as px
import torch
from torchvision.transforms import transforms
import codecs, json 
from skimage import data

"""
Improvements:
Add in the ability to toss in your own image and create the image
mask - maybe do this in a separate tab?_
Click which model to show how shitty old ones are
"""

def get_prediction_fig(sat_image, unet_weight, mask_size=420):
	sat_image = torch.Tensor(sat_image)
	if use_gpu:
		sat_image = sat_image.cuda()
		torch.cuda.empty_cache()
	
	unet_model = landpy.UNet(3,7)
	if use_gpu:
		unet_model = unet_model.cuda()
		unet_model.load_state_dict(torch.load(unet_weight))
	else:
		unet_model.load_state_dict(torch.load(unet_weight,
			map_location=torch.device('cpu')))
	unet_model.eval()

	predictions = unet_model(sat_image)
	soft_max_output = torch.nn.LogSoftmax(dim=1)(predictions)

	if use_gpu:
		soft_max_output = soft_max_output.cpu()
		
	numpy_output = soft_max_output.data.numpy()
	final_prediction = np.argmax(numpy_output,axis=1)
	prediction_img = landpy.construct_image(final_prediction[0], mask_size)
	prediction_array = np.transpose(prediction_img, (1,2,0))

	prediction_fig = px.imshow(prediction_array)
	prediction_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
	prediction_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

	return prediction_fig

def generate_actual_images_data(sat_image, labels):
	sat_array = np.transpose(sat_image[0].data.numpy(), (1,2,0))
	mask_array = np.transpose(labels[0].data.numpy(), (1,2,0))
	# prediction_array = np.transpose(prediction_img, (1,2,0))

	sat_image_fig = px.imshow(sat_array)
	sat_image_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
	sat_image_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

	sat_mask_fig = px.imshow(mask_array)
	sat_mask_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
	sat_mask_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

	#SERIALIZE NUMPY ARRAY
	b = sat_image.tolist()
	file_path = "path.json"
	json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

	return sat_image_fig, sat_mask_fig, file_path

def execute(data_loader, mask_size, unet_weights):

	beg_sat_image, beg_labels, beg_class_labels = next(iter(data_loader))

	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	
	# """
	# Front End of App Layout
	# """
	app.layout = html.Div([

		# html.Div([html.H2(html.U("Land Use Dashboard"))],
		# 	style={'textAlign': "center",
		# 	'backgroundColor': 'rgb(250,250,250)',
		# 	'color': 'black', 
		# 	'margin-bottom': '10px'}),

		html.Div([
			# Actual Image Div:
			html.Div([
				html.Div([html.H4("Satellite Image")],
					style={'textAlign':'center'}),
				dcc.Graph(id='sat-image')
				],
				style={'display': 'inline-block',
					'height': '100%', 'width': '33%',
					}
					),

			#Actual Mask Mask
			html.Div([
				html.Div([html.H4("Actual Mask")],
					style={'textAlign':'center'}),
				dcc.Graph(id='sat-mask')
				],
				style={'display': 'inline-block',
					'height': '100%', 'width': '33%',
					}
					),

			# Predicted Mask
			html.Div([
				html.Div([html.H4("Predicted Mask")],
					style={'textAlign':'center'}),
				dcc.Graph(id='prediction-mask'),
				html.Div(
					[dcc.Dropdown(id='unet_weights',options = unet_weights,
						value="weights/weights_optimal.pt")],
					style={'margin-top': '5px', 'margin-left': '120px',
					'width': '50%'}),
			  ],
			  style={'display': 'inline-block',
			      'height': '100%', 'width': '33%',
			      }
			      ),
			],
			style={'height': '50%'}),

		html.Div([html.Button('Click for Image', id='button', n_clicks=0)],
			style={'textAlign': "center", 'fontWeight': 'bold', 
			'padding': '0px 0px 200px 0px'}),

		html.Div(id='sat-image-array', style={'display': 'none'})
		])


	@app.callback([
		Output("sat-image", "figure"),
		Output("sat-mask", "figure"),
		Output("sat-image-array", 'children')],
		[
		Input('button', 'n_clicks')
		])
	def get_clicked(n_clicks):
		if n_clicks == 0:
			sat_image_fig, sat_mask_fig, file_path = generate_actual_images_data(beg_sat_image, beg_labels)
			return (sat_image_fig, sat_mask_fig, file_path)
		else:
			sat_image, labels, class_labels = next(iter(data_loader))
			sat_image_fig, sat_mask_fig, file_path = generate_actual_images_data(sat_image, labels)
			return (sat_image_fig, sat_mask_fig, file_path)
		
	@app.callback(
		Output("prediction-mask", "figure"),
		[Input('unet_weights', 'value'),
		Input("sat-image-array", 'children')
		])
	def update_prediction_figure(unet_weight, file_path):
		if not file_path:
			prediction_fig = get_prediction_fig(beg_sat_image, unet_weight)
			return prediction_fig
		
		obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
		b_new = json.loads(obj_text)
		sat_image = np.array(b_new)

		prediction_fig = get_prediction_fig(sat_image, unet_weight)
		return prediction_fig



	app.title = "Land Use Classification Dashboard"
	app.run_server(debug=True)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', type=str,
						help='Directory that contains images and masks')
	parser.add_argument('--model-path', type=str,
						help='File path that contains model weights')
	args = parser.parse_args()

	use_gpu = torch.cuda.is_available()
	t = transforms.Compose([
		transforms.Resize(612),
		transforms.ToTensor(),
		])

	t2 = transforms.Compose([
	transforms.Resize(420),
	transforms.ToTensor(),
	])

	data_set = landpy.MyDataLoader(args.data_dir, 420,
		image_transforms=t, mask_transforms=t2)

	train_loader, test_loader = landpy.create_data_loaders(
		data_set, 0.8, batch_size=1)

	unet_weights = glob.glob(args.model_path + "/*.pt")
	weight_dict = {
		'weights_jan_7.pt': "4th Best Model",
		'weights_jan_8.pt': "3rd Best Model",
		'weights_jan_9.pt': "2nd Best Model",
		'weights_optimal.pt': "Optimal Model"
	}
	unet_weights = [{'label': weight_dict[weight.split("/")[-1]], 'value': weight} for i, weight in enumerate(unet_weights)]

	execute(train_loader, mask_size=420, unet_weights=unet_weights)









