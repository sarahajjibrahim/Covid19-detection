# Covid-19 Detection using Segmentation, Deep Learning, Machine Learning 
Experiment - Segmentation:
1) Download the Normal and Covid data without or without masks from [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
2) Copy the folders (COVID, Normal) under \Experiments\data\unsegmented\
3) Test segmentation models (flood_segmentation.py, unet_segmentation.py or kmeans_segmentation.py ) 
For training U-net segmentation model on masks:
1) Download the CXR_png, masks and test folders from [Lung segmentation from Chest X-Ray dataset](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset)
2) Copy the folders (CXR_png, masks, test) under \Experiments\data\Lung Segmentation\ 

Experiment - Classification:
1) Segmented and normal images are fed as input
2) We test several DL and ML models to classify images and obtain results (classification.py)
 
WebApp (Using flask) - Single image classification using CNN:
1) On spyder, run app.py
2) Go to 127.0.0.1:5000/
3) Choose an x-ray image
4) To check the behaviour of model against filters or noise, choose a certain filter/noise
5) Submit x-ray and filter to display results


