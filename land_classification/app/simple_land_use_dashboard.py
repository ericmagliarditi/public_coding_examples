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

"""
Improvements:
Add in the ability to toss in your own image and create the image
mask - maybe do this in a separate tab?_
"""

def execute(data_loader, model, mask_size):

	sat_image, labels, class_labels = next(iter(data_loader))

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
				dcc.Graph(id='prediction-mask')
				
			  ],
			  style={'display': 'inline-block',
			      'height': '100%', 'width': '33%',
			      }
			      ),
			],
			style={'height': '50%'}),
		html.Div([html.Button('Click for Image', id='button', n_clicks=0)],
			style={'textAlign': "center",
			'padding': '0px 0px 0px 0px'})
		])

	@app.callback([
		Output("sat-image", "figure"),
		Output("sat-mask", "figure"),
		Output("prediction-mask", "figure")
		],
		[Input('button', 'n_clicks')])
	def get_clicked(n_clicks):
		sat_image, labels, class_labels = next(iter(data_loader))

		if use_gpu:
			sat_image = sat_image.cuda()
			torch.cuda.empty_cache()

		predictions = unet_model(sat_image)
		soft_max_output = torch.nn.LogSoftmax(dim=1)(predictions)

		if use_gpu:
			soft_max_output = soft_max_output.cpu()
			
		numpy_output = soft_max_output.data.numpy()
		final_prediction = np.argmax(numpy_output,axis=1)
		prediction_img = landpy.construct_image(final_prediction[0], mask_size)

		sat_array = np.transpose(sat_image[0].data.numpy(), (1,2,0))
		mask_array = np.transpose(labels[0].data.numpy(), (1,2,0))
		prediction_array = np.transpose(prediction_img, (1,2,0))

		sat_image_fig = px.imshow(sat_array)
		sat_image_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
		sat_image_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

		sat_mask_fig = px.imshow(mask_array)
		sat_mask_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
		sat_mask_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

		prediction_fig = px.imshow(prediction_array)
		prediction_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
		prediction_fig.update_layout(margin=dict(l=20,r=20,b=0,t=0))

		return sat_image_fig, sat_mask_fig, prediction_fig

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

	unet_model = landpy.UNet(3,7)
	if use_gpu:
		unet_model = unet_model.cuda()
		unet_model.load_state_dict(torch.load(args.model_path))
	else:
		unet_model.load_state_dict(torch.load(args.model_path,
			map_location=torch.device('cpu')))

	unet_model.eval()

	execute(train_loader, unet_model, mask_size=420)