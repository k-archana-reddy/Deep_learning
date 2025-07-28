from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Predict.html", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),
		     path("LoadDataset", views.LoadDataset, name="LoadDataset"),
		     path("TrainModel", views.TrainModel, name="TrainModel"),
		     
		      ]